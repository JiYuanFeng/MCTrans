import torch
import torch.nn as nn
from torch.nn.init import normal_

from mmcv.cnn import ConvModule

from ..builder import CENTERS
from ..ops.modules import MSDeformAttn
from ..trans.transformer import DSALayer, DSA
from ..trans.utils import build_position_encoding, NestedTensor


@CENTERS.register_module()
class MCTrans(nn.Module):
    def __init__(self,
                 d_model=240,
                 nhead=8,
                 d_ffn=1024,
                 dropout=0.1,
                 act="relu",
                 n_points=4,
                 n_levels=3,
                 n_sa_layers=6,
                 in_channles=[64, 64, 128, 256, 512],
                 proj_idxs=(2, 3, 4),

                 ):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.n_levels = n_levels

        self.proj_idxs = proj_idxs
        self.projs = nn.ModuleList()
        for idx in self.proj_idxs:
            self.projs.append(ConvModule(in_channles[idx],
                                         d_model,
                                         kernel_size=3,
                                         padding=1,
                                         conv_cfg=dict(type="Conv"),
                                         norm_cfg=dict(type='BN'),
                                         act_cfg=dict(type='ReLU')
                                         ))

        dsa_layer = DSALayer(d_model=d_model,
                             d_ffn=d_ffn,
                             dropout=dropout,
                             activation=act,
                             n_levels=n_levels,
                             n_heads=nhead,
                             n_points=n_points)

        self.dsa = DSA(att_layer=dsa_layer,
                       n_layers=n_sa_layers)

        self.level_embed = nn.Parameter(torch.Tensor(n_levels, d_model))
        self.position_embedding = build_position_encoding(position_embedding="sine", hidden_dim=d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def projection(self, feats):
        pos = []
        masks = []
        cnn_feats = []
        tran_feats = []

        for idx, feats in enumerate(feats):
            if idx not in self.proj_idxs:
                cnn_feats.append(feats)
            else:
                n, c, h, w = feats.shape
                mask = torch.zeros((n, h, w)).to(torch.bool).to(feats.device)
                nested_feats = NestedTensor(feats, mask)
                masks.append(mask)
                pos.append(self.position_embedding(nested_feats).to(nested_feats.tensors.dtype))
                tran_feats.append(feats)

        for idx, proj in enumerate(self.projs):
            tran_feats[idx] = proj(tran_feats[idx])

        return cnn_feats, tran_feats, pos, masks

    def forward(self, x):
        # project and prepare for the input
        cnn_feats, trans_feats, pos_embs, masks = self.projection(x)
        # dsa
        features_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        feature_shapes = []
        spatial_shapes = []
        for lvl, (feature, mask, pos_embed) in enumerate(zip(trans_feats, masks, pos_embs)):
            bs, c, h, w = feature.shape
            spatial_shapes.append((h, w))
            feature_shapes.append(feature.shape)

            feature = feature.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)

            features_flatten.append(feature)
            mask_flatten.append(mask)

        features_flatten = torch.cat(features_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=features_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # self att
        feats = self.dsa(features_flatten,
                         spatial_shapes,
                         level_start_index,
                         valid_ratios,
                         lvl_pos_embed_flatten,
                         mask_flatten)
        # recover
        out = []
        features = feats.split(spatial_shapes.prod(1).tolist(), dim=1)
        for idx, (feats, ori_shape) in enumerate(zip(features, spatial_shapes)):
            out.append(feats.transpose(1, 2).reshape(feature_shapes[idx]))

        cnn_feats.extend(out)
        return cnn_feats

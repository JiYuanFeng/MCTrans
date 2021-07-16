import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import HEADS, build_losses
from ...data.transforms import build_transforms


@HEADS.register_module()
class BasicSegHead(nn.Module):
    def __init__(self, in_channels, num_classes, kernel_size=1, post_trans=None, losses=None):
        super(BasicSegHead, self).__init__()
        self.head = nn.Conv2d(in_channels=in_channels, out_channels=num_classes, kernel_size=kernel_size)
        self.post_trans = build_transforms(post_trans)
        self.losses = build_losses(losses)

    def forward_train(self, inputs, seg_label, **kwargs):
        logits = self.head(inputs)
        losses = dict()
        for _loss in self.losses:
            losses[_loss.__class__.__name__] = _loss(logits, seg_label)
        return losses

    def forward_test(self, inputs, **kwargs):
        logits = self.head(inputs)
        preds = self.post_trans(logits)
        return preds

    def init_weights(self):
        pass
        # normal_init(self.head, mean=0, std=0.01)

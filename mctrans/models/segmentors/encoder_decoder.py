from torch import nn

from .base import BaseSegmentor
from ..builder import build_network, build_losses, MODEL, build_encoder, build_decoder, build_head, build_center
from ...data.transforms.utils import resize
from ...metrics import build_metrics


@MODEL.register_module()
class EncoderDecoder(BaseSegmentor):
    def __init__(self,
                 encoder,
                 decoder,
                 seg_head,
                 center=None,
                 aux_head=None,
                 pretrained=None):
        super(EncoderDecoder, self).__init__()
        self.encoder = build_encoder(encoder)
        self.decoder = build_decoder(decoder)
        self.seg_head = build_head(seg_head)

        if center is not None:
            self.center = build_center(center)
        if aux_head is not None:
            self.aux_head = build_head(aux_head)

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        pass
        super(EncoderDecoder, self).init_weights(pretrained)
        self.encoder.init_weights(pretrained=pretrained)
        self.decoder.init_weights()
        self.seg_head.init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.aux_head, nn.ModuleList):
                for aux_head in self.aux_head:
                    aux_head.init_weights()
            else:
                self.aux_head.init_weights()

    def extract_feat(self, img):
        x = self.encoder(img)
        if self.with_center:
            x = self.center(x)
        return x

    def encode_decode(self, img, img_metas, rescale=True):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        pred = self._decode_head_forward(x, img_metas, return_loss=False)
        #TODO shold evaluate on more dataset
        if rescale:
            if not hasattr(img_metas[0], "height"):
                re_size = img_metas[0]["spatial_shape"][:2]
            else:
                re_size = (img_metas[0]['height'], img_metas[0]['width'])
            pred = resize(
                pred,
                size=re_size,
                mode='nearest',
                align_corners=None,
                warning=False)
        return pred

    def _decode_head_forward(self, x, seg_label=None, return_loss=False):
        x = self.decoder(x)
        if return_loss:
            return self.seg_head.forward_train(x, seg_label)
        else:
            return self.seg_head.forward_test(x)

    def _auxiliary_head_forward(self, x, seg_label, return_loss=True):
        return self.aux_head.forward_train(x, seg_label)

    def forward_train(self, img, img_metas, seg_label, **kwargs):
        # the img_metas may useful in other framework
        x = self.extract_feat(img)

        losses = dict()
        loss_decode = self._decode_head_forward(x, seg_label, return_loss=True)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward(x, seg_label, return_loss=True)
            losses.update(loss_aux)

        return losses

    def forward_test(self, img, img_metas, rescale=True, **kwargs):
        # TODO support sliding window evaluator
        pred = self.encode_decode(img, img_metas, rescale)
        pred = pred.cpu().numpy()
        # unravel batch dim
        pred = list(pred)
        return pred

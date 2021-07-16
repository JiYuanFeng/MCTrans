from ..builder import LOSSES
from monai.losses import DiceLoss, DiceCELoss, FocalLoss, DiceFocalLoss


class DiceLoss(DiceLoss):
    def __init__(self, loss_weight=1.0, **kwargs):
        self.loss_weight = loss_weight
        super(DiceLoss, self).__init__(**kwargs)

    def forward(self, input, target):
        loss = self.loss_weight * super().forward(input=input, target=target)
        return loss


class DiceCELoss(DiceCELoss):
    def __init__(self, loss_weight=1.0, **kwargs):
        self.loss_weight = loss_weight
        super(DiceCELoss, self).__init__(**kwargs)

    def forward(self, input, target):
        loss = self.loss_weight * super().forward(input=input, target=target)
        return loss


LOSSES.register_module(name="DiceLoss", module=DiceLoss)
LOSSES.register_module(name="FocalLoss", module=FocalLoss)

LOSSES.register_module(name="DiceCELoss", module=DiceCELoss)
LOSSES.register_module(name="DiceFocalLoss", module=DiceFocalLoss)

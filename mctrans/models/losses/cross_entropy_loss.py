import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None):
    """Calculate the CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The gt label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: The calculated loss
    """
    # element-wise losses
    loss = F.cross_entropy(pred, label, reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def soft_cross_entropy(pred,
                       label,
                       weight=None,
                       reduction='mean',
                       avg_factor=None):
    """Calculate the Soft CrossEntropy loss. The label can be float.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The gt label of the prediction with shape (N, C).
            When using "mixup", the label can be float.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: The calculated loss
    """
    # element-wise losses
    loss = -label * F.log_softmax(pred, dim=-1)
    loss = loss.sum(dim=-1)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None):
    """Calculate the binary CrossEntropy loss with logits.

    Args:
        pred (torch.Tensor): The prediction with shape (N, *).
        label (torch.Tensor): The gt label with shape (N, *).
        weight (torch.Tensor, optional): Element-wise weight of loss with shape
             (N, ). Defaults to None.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". If reduction is 'none' , loss
             is same shape as pred and label. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: The calculated loss
    """
    assert pred.dim() == label.dim()

    loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        assert weight.dim() == 1
        weight = weight.float()
        if pred.dim() > 1:
            weight = weight.reshape(-1, 1)
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss

@LOSSES.register_module()
class CELoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(CELoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

        self.cls_criterion = cross_entropy

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        n_pred_ch, n_target_ch = cls_score.shape[1], label.shape[1]
        if n_pred_ch == n_target_ch:
            # target is in the one-hot format, convert to BH[WD] format to calculate ce loss
            label = torch.argmax(label, dim=1)
        else:
            label = torch.squeeze(label, dim=1)
        label = label.long()
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    """Cross entropy loss.

    Args:
        use_sigmoid (bool): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_soft (bool): Whether to use the soft version of CrossEntropyLoss.
            Defaults to False.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float):  Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 sigmoid=False,
                 softmax=False,
                 reduction='mean',
                 loss_weight=1.0):
        super(CrossEntropyLoss, self).__init__()
        self.use_sigmoid = sigmoid
        self.use_soft = softmax
        assert not (
                self.use_soft and self.use_sigmoid
        ), 'use_sigmoid and use_soft could not be set simultaneously'

        self.reduction = reduction
        self.loss_weight = loss_weight

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_soft:
            self.cls_criterion = soft_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        n_pred_ch, n_target_ch = cls_score.shape[1], label.shape[1]
        if n_pred_ch == n_target_ch:
            label = torch.argmax(label, dim=1)
        else:
            label = torch.squeeze(label, dim=1)
        label = label.long()

        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls


@LOSSES.register_module()
class MCTransAuxLoss(CrossEntropyLoss):
    def __init__(self,**kwargs):
        super(MCTransAuxLoss, self).__init__(**kwargs)

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        #To one hot
        num_classes = cls_score.shape[1]
        one_hot = []
        for l in label:
            one_hot.append(self.one_hot(torch.unique(l), num_classes=num_classes).sum(dim=0))
        label = torch.stack(one_hot)

        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls

    def one_hot(self, input, num_classes, dtype=torch.float):
        assert input.dim() > 0, "input should have dim of 1 or more."

        # if 1D, add singelton dim at the end
        if input.dim() == 1:
            input = input.view(-1, 1)

        sh = list(input.shape)

        assert sh[1] == 1, "labels should have a channel with length equals to one."
        sh[1] = num_classes

        o = torch.zeros(size=sh, dtype=dtype, device=input.device)
        labels = o.scatter_(dim=1, index=input.long(), value=1)

        return labels
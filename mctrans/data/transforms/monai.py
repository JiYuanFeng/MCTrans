from monai.transforms import RandSpatialCropd, SpatialPadd, CropForegroundd, RandCropByPosNegLabeld, Transform
from monai.transforms import Resized, Spacingd, Orientationd, RandRotated, RandZoomd, RandAxisFlipd, RandAffined
from monai.transforms import NormalizeIntensityd, ScaleIntensityRanged, ScaleIntensityd
from monai.transforms import AddChanneld, AsChannelFirstd, MapLabelValued, Lambdad, ToTensord
from monai.transforms import LoadImaged, RemoveRepeatedChanneld
from monai.transforms import Activations, AsDiscrete

import numpy as np
from ..builder import TRANSFORMS

# cropped
TRANSFORMS.register_module(name="RandCrop", module=RandSpatialCropd)
TRANSFORMS.register_module(name="SpatialPad", module=SpatialPadd)
TRANSFORMS.register_module(name="CropForeground", module=CropForegroundd)
TRANSFORMS.register_module(name="RandCropByPosNegLabel", module=RandCropByPosNegLabeld)
# spatial
TRANSFORMS.register_module(name='Resize', module=Resized)
TRANSFORMS.register_module(name="Spacing", module=Spacingd)
TRANSFORMS.register_module(name="Orientation", module=Orientationd)
TRANSFORMS.register_module(name="RandRotate", module=RandRotated)
TRANSFORMS.register_module(name="RandZoom", module=RandZoomd)
TRANSFORMS.register_module(name="RandAxisFlip", module=RandAxisFlipd)
TRANSFORMS.register_module(name="RandAffine", module=RandAffined)
# intensity
TRANSFORMS.register_module(name='NormalizeIntensity', module=NormalizeIntensityd)
TRANSFORMS.register_module(name='ScaleIntensityRange', module=ScaleIntensityRanged)
TRANSFORMS.register_module(name='ScaleIntensity', module=ScaleIntensityd)
# utility
TRANSFORMS.register_module(name="AddChannel", module=AddChanneld)
TRANSFORMS.register_module(name="AsChannelFirst", module=AsChannelFirstd)
TRANSFORMS.register_module(name="MapLabelValue", module=MapLabelValued)
TRANSFORMS.register_module(name="Lambda", module=Lambdad)
TRANSFORMS.register_module(name="ToTensor", module=ToTensord)
# io
TRANSFORMS.register_module(name="LoadImage", module=LoadImaged)
TRANSFORMS.register_module(name="RemoveRepeatedChannel", module=RemoveRepeatedChanneld)
# post-process
TRANSFORMS.register_module(name="Activations", module=Activations)
TRANSFORMS.register_module(name="AsDiscrete", module=AsDiscrete)


@TRANSFORMS.register_module()
class RandMirror(Transform):
    """ Mirror the data randomly along each specified axis according to the probability.

    Args:
        axis(None or int or tuple of ints): Along which axis to flip.
        prob(float): Probability of flipping.
    """

    def __init__(self, axis=(0, 1, 2), prob=0.5, image_key='img', label_key='seg_label'):
        self.axis = axis
        self.prob = prob
        self.label_key = label_key
        self.image_key = image_key

    def __call__(self, data):
        data = dict(data)
        image = data[self.image_key]
        seg_label = data[self.label_key] if self.label_key else None
        image, seg_label = self.augment_mirroring(image, seg_label)

        data[self.image_key] = image

        if self.label_key is not None:
            data[self.label_key] = seg_label

        return data

    def augment_mirroring(self, image, seg_label=None):
        if (len(image.shape) != 3) and (len(image.shape) != 4):
            raise Exception(
                "Invalid dimension for sample_data and sample_seg. sample_data and sample_seg should be either "
                "[channels, x, y] or [channels, x, y, z]")
        if 0 in self.axis and np.random.uniform() < self.prob:
            image[:, :] = image[:, ::-1]
            if seg_label is not None:
                seg_label[:, :] = seg_label[:, ::-1]
        if 1 in self.axis and np.random.uniform() < self.prob:
            image[:, :, :] = image[:, :, ::-1]
            if seg_label is not None:
                seg_label[:, :, :] = seg_label[:, :, ::-1]
        if 2 in self.axis and len(image.shape) == 4:
            if np.random.uniform() < self.prob:
                image[:, :, :, :] = image[:, :, :, ::-1]
                if seg_label is not None:
                    seg_label[:, :, :, :] = seg_label[:, :, :, ::-1]
        return image, seg_label
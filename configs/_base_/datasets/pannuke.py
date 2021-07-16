dataset_type = 'PanNukeDataset'
patch_size = [256, 256]

keyword = ["img", "seg_label"]

train_transforms = [
    dict(type="LoadImage",
         keys=keyword,
         meta_key_postfix="metas"),
    dict(type="AsChannelFirst",
         keys=keyword[0]),
    dict(type="AddChannel",
         keys=keyword[1]),
    dict(type='Resize',
         keys=keyword,
         spatial_size=patch_size,
         mode=("bilinear", "nearest")),
    dict(type="ScaleIntensity",
         keys=keyword[0]),
    dict(type='RandMirror'),
    dict(type='ToTensor',
         keys=keyword)
]

test_transforms = [
    dict(type="LoadImage",
         keys=keyword,
         meta_key_postfix="metas"),
    dict(type="AsChannelFirst",
         keys=keyword[0]),
    dict(type="AddChannel",
         keys=keyword[1]),
    dict(type='Resize',
         keys=keyword,
         spatial_size=patch_size,
         mode=("bilinear", "nearest")),
    dict(type="ScaleIntensity",
         keys=keyword[0]),
    dict(type='ToTensor',
         keys=keyword)
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        img_dirs=["data/medical/pannuke/split-images-npy/0",
                  "data/medical/pannuke/split-images-npy/1"],
        img_suffix=".npy",
        label_dirs=["data/medical/pannuke/split-masks-npy/0",
                    "data/medical/pannuke/split-masks-npy/1"],
        label_suffix=".npy",
        phase="train",
        transforms=train_transforms,
    ),
    val=dict(
        type=dataset_type,
        img_dirs=["data/medical/pannuke/split-images-npy/2"],
        img_suffix=".npy",
        label_dirs=["data/medical/pannuke/split-masks-npy/2"],
        label_suffix=".npy",
        phase="val",
        transforms=test_transforms
    ),
)
#support data with different formats
#
# data = dict(
#     samples_per_gpu=32,
#     workers_per_gpu=8,
#     train=dict(
#         type=dataset_type,
#         img_dirs=["data/medical/pannuke/split-images/0",
#                   "data/medical/pannuke/split-images/1"],
#         img_suffix=".png",
#         label_dirs=["data/medical/pannuke/split-masks/0",
#                     "data/medical/pannuke/split-masks/1"],
#         label_suffix=".png",
#         phase="train",
#         transforms=train_transforms,
#     ),
#     val=dict(
#         type=dataset_type,
#         img_dirs=["data/medical/pannuke/split-images/2"],
#         img_suffix=".png",
#         label_dirs=["data/medical/pannuke/split-masks/2"],
#         label_suffix=".png",
#         phase="val",
#         transforms=test_transforms
#     ),
# )
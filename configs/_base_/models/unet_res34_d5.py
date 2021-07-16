model = dict(
    type='EncoderDecoder',
    pretrained=None,
    encoder=dict(
        type="ResNet",
        depth=34,
        in_channels=3),
    decoder=dict(
        type="UNetDecoder",
        in_channels=[64, 64, 128, 256, 512],
    ),
    seg_head=dict(
        type="BasicSegHead",
        in_channels=64,
        num_classes=6,
        post_trans=[dict(type="Activations", softmax=True),
                    dict(type="AsDiscrete", argmax=True)],
        losses=[dict(type="DiceCELoss", softmax=True, to_onehot_y=True),
                dict(type="FocalLoss", to_onehot_y=True)])
)

model = dict(
    type='EncoderDecoder',
    pretrained=None,
    encoder=dict(
        type="VGG",
        in_channel=3,
        init_channels=32,
        num_blocks=2),
    decoder=dict(
        type="UNetDecoder",
        in_channels=[32, 64, 128, 256, 512],
        att=True
    ),
    seg_head=dict(
        type="BasicSegHead",
        in_channels=32,
        num_classes=6,
        post_trans=[dict(type="Activations", softmax=True),
                    dict(type="AsDiscrete", argmax=True)],
        losses=[dict(type="DiceLoss", softmax=True, to_onehot_y=True),
                dict(type="CrossEntropyLoss"),
                dict(type="FLoss")]
    )
)

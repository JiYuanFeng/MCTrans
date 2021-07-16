model = dict(
    type='EncoderDecoder',
    pretrained=None,
    encoder=dict(
        type="ResNet",
        depth=34,
        in_channels=3),
    center=dict(
        type="MCTrans",
        d_model=128,
        nhead=8,
        d_ffn=512,
        dropout=0.1,
        act="relu",
        n_levels=3,
        n_points=4,
        n_sa_layers=6),
    decoder=dict(
        type="UNetDecoder",
        in_channels=[64, 64, 128, 128, 128],
    ),
    seg_head=dict(
        type="BasicSegHead",
        in_channels=64,
        num_classes=6,
        post_trans=[dict(type="Activations", softmax=True),
                    dict(type="AsDiscrete", argmax=True)],
        losses=[dict(type="DiceCELoss", softmax=True, to_onehot_y=True),
                dict(type="FocalLoss", to_onehot_y=True)]),
    aux_head=dict(
        type="MCTransAuxHead",
        d_model=128,
        d_ffn=512,
        act="relu",
        num_classes=6,
        in_channles=[64, 64, 128, 128, 128],
        losses=[dict(type="MCTransAuxLoss", sigmoid=True, loss_weight=0.1)]),
)

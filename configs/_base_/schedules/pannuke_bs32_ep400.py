# optimizer
optimizer = dict(type='Adam', lr=0.0003, betas=[0.9, 0.99])
optimizer_config = dict()
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.0)
# runtime settings
max_epochs = 400
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)
checkpoint_config = dict(by_epoch=True, interval=5)
evaluation = dict(interval=1, save_best="mDice")
_base_ = [
    '../_base_/models/attunet_vgg32_d5.py', '../_base_/datasets/pannuke.py',
    '../_base_/default_runtime.py', '../_base_/schedules/pannuke_bs32_ep400.py'
]

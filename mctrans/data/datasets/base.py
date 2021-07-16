import os
import os.path as osp
from functools import reduce
from collections import OrderedDict

import mmcv
import monai
from mmcv import print_log
import numpy as np

from prettytable import PrettyTable
from sklearn.model_selection import KFold
from torch.utils.data import Dataset

from ..builder import DATASETS
from ..transforms import build_transforms
from ...metrics.base import eval_metrics
from ...utils import get_root_logger


@DATASETS.register_module()
class BaseDataset(Dataset):
    """ Custom Dataset for medical image segmentation"""
    CLASSES = None
    PALETTE = None

    def __init__(self,
                 transforms,
                 img_dirs,
                 img_suffix=".jpg",
                 label_dirs=None,
                 label_suffix=".png",
                 phase=None,
                 cross_valid=False,
                 fold_idx=0,
                 fold_nums=5,
                 data_root=None,
                 ignore_index=None,
                 binnary_label=False,
                 exclude_backgroud=True,
                 label_map=None
                 ):
        self.transforms = build_transforms(transforms)

        self.img_dirs = img_dirs if isinstance(img_dirs, (list, tuple)) else [img_dirs]
        self.label_dirs = label_dirs if isinstance(label_dirs, (list, tuple)) else [label_dirs]
        self.img_suffix = img_suffix
        self.label_suffix = label_suffix
        self.phase = phase
        self.cross_valid = cross_valid
        self.fold_idx = fold_idx
        self.fold_nums = fold_nums
        self.data_root = data_root
        self.label_map = label_map
        self.binary_label = binnary_label
        self.ignore_index = ignore_index
        self.exclude_backgroud = exclude_backgroud
        self.data_list = self.generate_data_list(self.img_dirs, self.img_suffix, self.label_dirs, self.label_suffix,
                                                 self.cross_valid, self.phase, self.fold_idx, self.fold_nums)

    def generate_data_list(self,
                           img_dirs,
                           img_suffix,
                           label_dirs,
                           label_suffix,
                           cross_valid=False,
                           phase="Train",
                           fold_idx=0,
                           fold_nums=5,
                           img_key="img",
                           label_key="seg_label"):

        if label_dirs is not None:
            assert len(img_dirs) == len(label_dirs)

        data_list = []
        for idx, img_dir in enumerate(img_dirs):
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                data_info = {}
                data_info[img_key] = osp.join(img_dir, img)
                if label_dirs is not None:
                    label = img.replace(img_suffix, label_suffix)
                    data_info[label_key] = osp.join(label_dirs[idx], label)
                data_list.append(data_info)

        if cross_valid:
            assert isinstance(fold_idx, int) and isinstance(fold_nums, int)
            splits = []
            kfold = KFold(n_splits=fold_nums, shuffle=True)

            for tr_idx, te_idx in kfold.split(data_list):
                splits.append(dict())
                splits[-1]['train'] = [item for idx, item in enumerate(data_list) if idx in tr_idx]
                splits[-1]['val'] = [item for idx, item in enumerate(data_list) if idx in te_idx]
            data_list = splits[fold_idx][phase]

        print_log("Phase {} : Loaded {} images".format(phase, len(data_list)), logger=get_root_logger())

        return data_list

    def get_gt_seg_maps(self):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        reader = monai.transforms.LoadImage(image_only=True)
        for img_info in self.data_list:
            seg_map = osp.join(img_info['seg_label'])
            # gt_seg_map = mmcv.imread(
            #     seg_map, flag='unchanged', backend='pillow')
            gt_seg_map = reader(seg_map)
            # binary the mask if need
            if self.binary_label:
                gt_seg_map[gt_seg_map > 0] = 255
            # modify if custom classes
            if self.label_map is not None:
                for old_id, new_id in self.label_map.items():
                    gt_seg_map[gt_seg_map == old_id] = new_id
            gt_seg_maps.append(gt_seg_map)

        return gt_seg_maps

    def evaluate(self,
                 results,
                 metric=['mDice', "mIoU", "mFscore", "mHd95"],
                 logger=None,
                 **kwargs):
        """Evaluate the dataset.
        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            dict[str, float]: Default metrics.
        """

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore', "mHd95"]
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps()

        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)

        ret_metrics = eval_metrics(
            results,
            gt_seg_maps,
            num_classes,
            self.ignore_index,
            metric)

        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # summary table
        # exclude some ignore idx
        if self.exclude_backgroud:
            for ret_metric, ret_metric_value in ret_metrics.items():
                ret_metrics[ret_metric] = ret_metric_value[1:]

        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(class_names)
            })

        if mmcv.is_list_of(results, str):
            for file_name in results:
                os.remove(file_name)
        return eval_results

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        data = self.transforms(data)
        return data

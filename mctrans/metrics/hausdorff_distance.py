from typing import Union, Optional

import mmcv
import numpy as np
import torch
from monai.metrics import get_mask_edges, get_surface_distance


def compute_hausdorff_distance(
    y_pred: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    include_background: bool = False,
    distance_metric: str = "euclidean",
    percentile: Optional[float] = None,
    directed: bool = False,
):
    """
    Compute the Hausdorff distance.

    Args:
        y_pred: input data to compute, typical segmentation model output.
            It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
            should be binarized.
        y: ground truth to compute mean the distance. It must be one-hot format and first dim is batch.
            The values should be binarized.
        include_background: whether to skip distance computation on the first channel of
            the predicted output. Defaults to ``False``.
        distance_metric: : [``"euclidean"``, ``"chessboard"``, ``"taxicab"``]
            the metric used to compute surface distance. Defaults to ``"euclidean"``.
        percentile: an optional float number between 0 and 100. If specified, the corresponding
            percentile of the Hausdorff Distance rather than the maximum result will be achieved.
            Defaults to ``None``.
        directed: whether to calculate directed Hausdorff distance. Defaults to ``False``.
    """
    if isinstance(y_pred, str):
        y_pred = torch.from_numpy(np.load(y_pred))
    else:
        y_pred = torch.from_numpy((y_pred))

    if isinstance(y, str):
        y = torch.from_numpy(
            mmcv.imread(y, flag='unchanged', backend='pillow'))
    else:
        y = torch.from_numpy(y)

    if isinstance(y, torch.Tensor):
        y = y.float()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.float()

    if y.shape != y_pred.shape:
        y_pred = torch.squeeze(y_pred, dim=0)

    batch_size, n_class = y_pred.shape[:2]
    hd = np.empty((batch_size, n_class))

    (edges_pred, edges_gt) = get_mask_edges(y_pred, y)
    distance_1 = compute_percent_hausdorff_distance(edges_pred, edges_gt, distance_metric, percentile)

    for b, c in np.ndindex(batch_size, n_class):
        (edges_pred, edges_gt) = get_mask_edges(y_pred[b, c], y[b, c])
        distance_1 = compute_percent_hausdorff_distance(edges_pred, edges_gt, distance_metric, percentile)
        if directed:
            hd[b, c] = distance_1
        else:
            distance_2 = compute_percent_hausdorff_distance(edges_gt, edges_pred, distance_metric, percentile)
            hd[b, c] = max(distance_1, distance_2)
    return torch.from_numpy(hd)



def compute_percent_hausdorff_distance(
    edges_pred: np.ndarray,
    edges_gt: np.ndarray,
    distance_metric: str = "euclidean",
    percentile: Optional[float] = None,
):
    """
    This function is used to compute the directed Hausdorff distance.
    """

    surface_distance = get_surface_distance(edges_pred, edges_gt, distance_metric=distance_metric)

    # for both pred and gt do not have foreground
    if surface_distance.shape == (0,):
        return np.nan

    if not percentile:
        return surface_distance.max()

    if 0 <= percentile <= 100:
        return np.percentile(surface_distance, percentile)
    raise ValueError(f"percentile should be a value between 0 and 100, get {percentile}.")


def total_hausdorff_distance(results,
                              gt_seg_maps,
                              ):
    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs

    for i in range(num_imgs):
        x= compute_hausdorff_distance(results[i], gt_seg_maps[i], percentile=95)
    return 0
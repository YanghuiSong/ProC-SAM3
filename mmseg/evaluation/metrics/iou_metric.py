# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
from PIL import Image
from prettytable import PrettyTable

from mmseg.registry import METRICS


@METRICS.register_module()
class IoUMetric(BaseMetric):
    """IoU evaluation metric.

    Args:
        ignore_index (int): Index that will be ignored in evaluation.
            Default: 255.
        iou_metrics (list[str] | str): Metrics to be calculated, the options
            includes 'mIoU', 'mDice' and 'mFscore'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        beta (int): Determines the weight of recall in the combined score.
            Default: 1.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        output_dir (str): The directory for output prediction. Defaults to
            None.
        format_only (bool): Only format result for results commit without
            perform evaluation. It is useful when you want to save the result
            to a specific format and submit it to the test server.
            Defaults to False.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    def __init__(self,
                 ignore_index: int = 255,
                 iou_metrics: List[str] = ['mIoU'],
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.metrics = iou_metrics
        self.nan_to_num = nan_to_num
        self.beta = beta
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        num_classes = len(self.dataset_meta['classes'])
        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()
            # format_only always for test dataset without ground truth
            if not self.format_only:
                label = data_sample['gt_sem_seg']['data'].squeeze().to(
                    pred_label)
                self.results.append(
                    self.intersect_and_union(pred_label, label, num_classes,
                                             self.ignore_index))
            # format_result
            if self.output_dir is not None:
                basename = osp.splitext(osp.basename(
                    data_sample['img_path']))[0]
                png_filename = osp.abspath(
                    osp.join(self.output_dir, f'{basename}.png'))
                output_mask = pred_label.cpu().numpy()
                # The index range of official ADE20k dataset is from 0 to 150.
                # But the index range of output is from 0 to 149.
                # That is because we set reduce_zero_label=True.
                if data_sample.get('reduce_zero_label', False):
                    output_mask = output_mask + 1
                output = Image.fromarray(output_mask.astype(np.uint8))
                output.save(png_filename)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()
        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
        results = tuple(zip(*results))
        assert len(results) == 4

        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_pred_label = sum(results[2])
        total_area_label = sum(results[3])
        ret_metrics = self.total_area_to_metrics(
            total_area_intersect, total_area_union, total_area_pred_label,
            total_area_label, self.metrics, self.nan_to_num, self.beta)

        class_names = self.dataset_meta['classes']

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        metrics = dict()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                metrics[key] = val
            else:
                metrics['m' + key] = val

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        return metrics

    @staticmethod
    def intersect_and_union(pred_label: torch.tensor, label: torch.tensor,
                            num_classes: int, ignore_index: int):
        """Calculate Intersection and Union.

        Args:
            pred_label (torch.tensor): Prediction segmentation map
                or predict result filename. The shape is (H, W).
            label (torch.tensor): Ground truth segmentation map
                or label filename. The shape is (H, W).
            num_classes (int): Number of categories.
            ignore_index (int): Index that will be ignored in evaluation.

        Returns:
            torch.Tensor: The intersection of prediction and ground truth
                histogram on all classes.
            torch.Tensor: The union of prediction and ground truth histogram on
                all classes.
            torch.Tensor: The prediction histogram on all classes.
            torch.Tensor: The ground truth histogram on all classes.
        """

        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect = torch.histc(
            intersect.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_pred_label = torch.histc(
            pred_label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_label = torch.histc(
            label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_union = area_pred_label + area_label - area_intersect
        return area_intersect, area_union, area_pred_label, area_label

    @staticmethod
    def total_area_to_metrics(total_area_intersect: np.ndarray,
                              total_area_union: np.ndarray,
                              total_area_pred_label: np.ndarray,
                              total_area_label: np.ndarray,
                              metrics: List[str] = ['mIoU'],
                              nan_to_num: Optional[int] = None,
                              beta: int = 1):
        """Calculate evaluation metrics
        Args:
            total_area_intersect (np.ndarray): The intersection of prediction
                and ground truth histogram on all classes.
            total_area_union (np.ndarray): The union of prediction and ground
                truth histogram on all classes.
            total_area_pred_label (np.ndarray): The prediction histogram on
                all classes.
            total_area_label (np.ndarray): The ground truth histogram on
                all classes.
            metrics (List[str] | str): Metrics to be evaluated, 'mIoU' and
                'mDice'.
            nan_to_num (int, optional): If specified, NaN values will be
                replaced by the numbers defined by the user. Default: None.
            beta (int): Determines the weight of recall in the combined score.
                Default: 1.
        Returns:
            Dict[str, np.ndarray]: per category evaluation metrics,
                shape (num_classes, ).
        """

        def f_score(precision, recall, beta=1):
            """calculate the f-score value.

            Args:
                precision (float | torch.Tensor): The precision value.
                recall (float | torch.Tensor): The recall value.
                beta (int): Determines the weight of recall in the combined
                    score. Default: 1.

            Returns:
                [torch.tensor]: The f-score value.
            """
            score = (1 + beta**2) * (precision * recall) / (
                (beta**2 * precision) + recall)
            return score

        if isinstance(metrics, str):
            metrics = [metrics]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metrics).issubset(set(allowed_metrics)):
            raise KeyError(f'metrics {metrics} is not supported')

        all_acc = total_area_intersect.sum() / total_area_label.sum()
        ret_metrics = OrderedDict({'aAcc': all_acc})
        for metric in metrics:
            if metric == 'mIoU':
                iou = total_area_intersect / total_area_union
                acc = total_area_intersect / total_area_label
                ret_metrics['IoU'] = iou
                ret_metrics['Acc'] = acc
            elif metric == 'mDice':
                dice = 2 * total_area_intersect / (
                    total_area_pred_label + total_area_label)
                acc = total_area_intersect / total_area_label
                ret_metrics['Dice'] = dice
                ret_metrics['Acc'] = acc
            elif metric == 'mFscore':
                precision = total_area_intersect / total_area_pred_label
                recall = total_area_intersect / total_area_label
                f_value = torch.tensor([
                    f_score(x[0], x[1], beta) for x in zip(precision, recall)
                ])
                ret_metrics['Fscore'] = f_value
                ret_metrics['Precision'] = precision
                ret_metrics['Recall'] = recall

        ret_metrics = {
            metric: value.numpy()
            for metric, value in ret_metrics.items()
        }
        if nan_to_num is not None:
            ret_metrics = OrderedDict({
                metric: np.nan_to_num(metric_value, nan=nan_to_num)
                for metric, metric_value in ret_metrics.items()
            })
        return ret_metrics


@METRICS.register_module()
class SegAPMetric(BaseMetric):
    """Segmentation AP evaluation metric (Average Precision at different IoU thresholds).

    Args:
        ignore_index (int): Index that will be ignored in evaluation.
            Default: 255.
        iou_thresholds (list[float]): List of IoU thresholds to calculate AP.
            Default: [0.5, 0.75] for AP50 and AP75.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    def __init__(self,
                 ignore_index: int = 255,
                 iou_thresholds: List[float] = [0.5, 0.75],
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.iou_thresholds = iou_thresholds

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()
            label = data_sample['gt_sem_seg']['data'].squeeze().to(pred_label)
            
            # Calculate AP for each IoU threshold
            ap_results = self.calculate_segmentation_ap(
                pred_label, label, self.iou_thresholds, len(self.dataset_meta['classes']), self.ignore_index
            )
            
            self.results.append(ap_results)

    def calculate_segmentation_ap(self, pred: torch.tensor, target: torch.tensor, 
                                 iou_thresholds: List[float], num_classes: int, 
                                 ignore_index: int) -> Dict[float, torch.tensor]:
        """Calculate segmentation AP at different IoU thresholds for each class.
        
        Args:
            pred: Predicted segmentation mask (H, W)
            target: Ground truth segmentation mask (H, W)
            iou_thresholds: List of IoU thresholds to calculate AP for
            num_classes: Number of classes
            ignore_index: Index to ignore in evaluation
            
        Returns:
            Dictionary mapping IoU threshold to AP per class
        """
        # Mask out ignored pixels
        mask = (target != ignore_index)
        pred_masked = pred[mask]
        target_masked = target[mask]

        ap_results = {}

        for iou_thresh in iou_thresholds:
            # Calculate IoU for each class
            class_ious = torch.zeros(num_classes, dtype=torch.float32)
            
            for cls_idx in range(num_classes):
                # Create binary masks for current class
                pred_cls = (pred_masked == cls_idx)
                target_cls = (target_masked == cls_idx)
                
                # Skip if class not present in ground truth
                if not target_cls.any():
                    class_ious[cls_idx] = float('nan')  # Mark as NaN for classes not present
                    continue
                
                # Calculate intersection and union for this class
                intersection = (pred_cls & target_cls).sum().float()
                union = (pred_cls | target_cls).sum().float()
                
                if union == 0:
                    class_ious[cls_idx] = float('nan')  # Mark as NaN
                else:
                    iou = intersection / union
                    # If IoU >= threshold, consider as correct detection
                    class_ious[cls_idx] = 1.0 if iou >= iou_thresh else 0.0

            ap_results[iou_thresh] = class_ious

        return ap_results

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics.
        """
        # Combine results across all images
        if not results:
            return {}

        # Initialize accumulators for each IoU threshold
        ap_accumulators = {}
        for iou_thresh in self.iou_thresholds:
            # Stack all class APs for this threshold across all images
            aps_per_class = torch.stack([result[iou_thresh] for result in results])
            
            # Calculate mean AP per class (ignoring NaN values)
            mean_aps = []
            num_classes = aps_per_class.shape[1]
            
            for cls_idx in range(num_classes):
                class_aps = aps_per_class[:, cls_idx]
                # Remove NaN values before computing mean
                valid_aps = class_aps[~torch.isnan(class_aps)]
                if len(valid_aps) > 0:
                    mean_aps.append(valid_aps.mean().item())
                else:
                    mean_aps.append(0.0)  # Default to 0 if no valid values
            
            # Calculate overall mean AP for this threshold
            mean_ap = np.nanmean(mean_aps)
            ap_accumulators[f'AP{int(iou_thresh * 100)}'] = round(mean_ap * 100, 2)

        return ap_accumulators


@METRICS.register_module()
class CombinedMetrics(BaseMetric):
    """Combined evaluation metric that computes both IoU and AP metrics.

    Args:
        ignore_index (int): Index that will be ignored in evaluation.
            Default: 255.
        iou_metrics (list[str] | str): Metrics to be calculated, the options
            includes 'mIoU', 'mDice' and 'mFscore'.
        iou_thresholds (list[float]): List of IoU thresholds to calculate AP.
            Default: [0.5, 0.75] for AP50 and AP75.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        beta (int): Determines the weight of recall in the combined score.
            Default: 1.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    def __init__(self,
                 ignore_index: int = 255,
                 iou_metrics: List[str] = ['mIoU'],
                 iou_thresholds: List[float] = [0.5, 0.75],
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.iou_metrics = iou_metrics
        self.iou_thresholds = iou_thresholds
        self.nan_to_num = nan_to_num
        self.beta = beta

        # Create separate metric instances
        self.iou_metric = IoUMetric(
            ignore_index=ignore_index,
            iou_metrics=iou_metrics,
            nan_to_num=nan_to_num,
            beta=beta,
            collect_device=collect_device
        )
        self.ap_metric = SegAPMetric(
            ignore_index=ignore_index,
            iou_thresholds=iou_thresholds,
            collect_device=collect_device
        )

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples for both IoU and AP metrics."""
        # Set dataset_meta for child metrics if parent has it
        if hasattr(self, 'dataset_meta') and self.dataset_meta is not None:
            self.iou_metric.dataset_meta = self.dataset_meta
            self.ap_metric.dataset_meta = self.dataset_meta
        
        # Process with IoU metric
        self.iou_metric.process(data_batch, data_samples)
        # Process with AP metric
        self.ap_metric.process(data_batch, data_samples)
        # Add dummy entry to prevent empty results warning
        self.results.append({'processed': True})

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute combined metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics combining both IoU and AP.
        """
        # We need to handle the fact that both metrics have processed the same data
        # So we compute metrics separately using their own internal results
        iou_results = self.iou_metric.compute_metrics(self.iou_metric.results)
        ap_results = self.ap_metric.compute_metrics(self.ap_metric.results)
        
        # Combine both sets of results
        combined_results = {**iou_results, **ap_results}
        return combined_results

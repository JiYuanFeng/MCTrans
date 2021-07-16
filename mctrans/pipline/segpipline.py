import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import build_runner, build_optimizer
from mmcv.runner import DistSamplerSeedHook, DistEvalHook, EvalHook

from mctrans.data import build_dataset, build_dataloader
from mctrans.models import build_model
from mctrans.utils import get_root_logger


class SegPipline(object):
    def __init__(self, cfg, distributed=False, validate=False):
        self.cfg = cfg
        self.distributed = distributed
        self.logger = get_root_logger(cfg.log_level)
        self.data_loaders = self._build_data_loader()
        self.model = self._build_model()
        # need to update
        self.optimizer = build_optimizer(self.model, cfg.optimizer)
        self.runner = self._build_runner()
        if validate:
            self._build_eval_hook()
        if cfg.resume_from:
            self.runner.resume(cfg.resume_from)
        elif cfg.load_from:
            self.runner.load_checkpoint(cfg.load_from)
        else:
            pass

    def _build_data_loader(self):
        dataset = [build_dataset(self.cfg.data.train)]
        if len(self.cfg.workflow) == 2:
            dataset.append(build_dataset(self.cfg.data.val))
        data_loaders = [
            build_dataloader(
                ds,
                self.cfg.data.samples_per_gpu,
                self.cfg.data.workers_per_gpu,
                len(self.cfg.gpu_ids),
                dist=self.distributed,
                seed=self.cfg.seed,
                drop_last=True) for ds in dataset]
        return data_loaders

    def _build_model(self):
        model = build_model(self.cfg.model)
        if self.distributed:
            find_unused_parameters = self.cfg.get('find_unused_parameters', True)
            # Sets the `find_unused_parameters` parameter in
            # torch.nn.parallel.DistributedDataParallel
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            model = MMDataParallel(
                model.cuda(self.cfg.gpu_ids[0]), device_ids=self.cfg.gpu_ids)
        return model

    def _build_eval_hook(self):
        val_dataset = build_dataset(self.cfg.data.val)
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=self.cfg.data.workers_per_gpu,
            dist=self.distributed,
            shuffle=False)
        eval_cfg = self.cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = self.cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if self.distributed else EvalHook
        self.runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    def _build_runner(self):
        runner = build_runner(
            cfg=self.cfg.runner,
            default_args=dict(
                model=self.model,
                batch_processor=None,
                optimizer=self.optimizer,
                work_dir=self.cfg.work_dir,
                logger=self.logger,
                meta=None))

        # register hooks
        runner.register_training_hooks(self.cfg.lr_config, self.cfg.optimizer_config,
                                       self.cfg.checkpoint_config, self.cfg.log_config,
                                       self.cfg.get('momentum_config', None))

        if self.distributed:
            runner.register_hook(DistSamplerSeedHook())

        return runner

    def _report_details(self):
        self.logger.info(self.model)

    def run(self):
        self.runner.run(self.data_loaders, self.cfg.workflow, self.cfg.max_epochs)

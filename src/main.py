# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------

import os
import sys
import argparse
import datetime
import glob
import json
import random
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np

from . import build_model_main

from util.get_param_dicts import get_param_dict
from util.logger import setup_logger, fprint
from util.slconfig import DictAction, SLConfig
from util.utils import ModelEma, BestMetricHolder
import util.misc as utils

from datasets.dataset import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch, evaluate_ap


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--config_file", "-c", type=str, required=True)
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file.",
    )

    # dataset parameters
    parser.add_argument("--dataset_file", default="coco")
    parser.add_argument(
        "--coco_path", type=str, default="/comp_robot/cv_public_dataset/COCO2017/"
    )
    parser.add_argument("--coco_panoptic_path", type=str)
    parser.add_argument("--remove_difficult", action="store_true")
    parser.add_argument("--fix_size", action="store_true")

    # training parameters
    parser.add_argument(
        "--output_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument("--note", default="", help="add some notes to the experiment")
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=42, type=int)

    # Resume training from the highest checkpoint present in the output_dir => will reuse LR + optimizer of the previous model
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    # Use checkpoint to initialize the model and optimizer, but not the LR (for finetuning)
    parser.add_argument("--pretrain_model_path", help="load from other checkpoint")
    # Layers where weights are kept (usually first layers that are less specialized)
    parser.add_argument("--finetune_ignore", type=str, nargs="+")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval_all", action="store_true", help="Evaluate model on every epochs available")

    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--find_unused_params", action="store_true")

    parser.add_argument("--save_results", action="store_true")
    parser.add_argument("--save_log", action="store_true")

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--local_rank", type=int, help="local rank for DistributedDataParallel"
    )
    parser.add_argument("--amp", action="store_true", help="Train with mixed precision")

    return parser


def main(args):
    utils.init_distributed_mode(args)
    time.sleep(args.rank * 0.02)

    # load cfg file and update the args
    print(f"Loading config file from {args.config_file}")
    cfg = SLConfig.fromfile(args.config_file)

    model_dir = Path(args.output_dir)
    config_path = model_dir / "config_cfg.py"
    info_path = model_dir / "info.txt"

    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if args.rank == 0:
        cfg.dump(f"{config_path}")
        with open(model_dir / "config_args_raw.json", "w") as f:
            json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError(f"Key {f} can used by args only")

    if args.use_wandb:
        is_eval = args.eval or args.eval_all
        import wandb
        run = wandb.init(
            project=f"{'eval' if is_eval else 'train'}-dino-primitives",
            name=f"{model_dir.name}_{datetime.date.today()}",
            config=cfg_dict,
            notes=model_dir.name,
        )
    else:
        run = None

    # update some new args temporally
    if not getattr(args, "use_ema", None):
        args.use_ema = False
    if not getattr(args, "debug", None):
        args.debug = False

    os.makedirs(model_dir, exist_ok=True)

    logger = setup_logger(
        output=f"{info_path}",
        distributed_rank=args.rank,
        color=False,
        name="detr",
    )
    logger.info(f"git:\n {utils.get_sha()}\n")
    logger.info(f"Command: {' '.join(sys.argv)}")
    if args.rank == 0:
        with open(model_dir / "config_args_all.json", "w") as f:
            json.dump(vars(args), f, indent=2)
        logger.info(f"Full config saved to {model_dir}/config_args_all.json")

    logger.info(f"world size: {args.world_size}")
    logger.info(f"rank: {args.rank}")
    logger.info(f"local_rank: {args.local_rank}")
    logger.info(f"args: {args}\n")

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model, criterion, postprocessors = build_model_main(args)
    wo_class_error = False
    model.to(device)

    # ema
    ema_m = ModelEma(model, args.ema_decay) if args.use_ema else None

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=args.find_unused_params
        )
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    logger.info(
        "params:\n"
        + json.dumps(
            {n: p.numel() for n, p in model.named_parameters() if p.requires_grad},
            indent=2,
        )
    )

    param_dicts = get_param_dict(args, model_without_ddp)

    optimizer = torch.optim.AdamW(
        param_dicts, lr=args.lr, weight_decay=args.weight_decay
    )

    ### DATASET ###
    dataset_train = build_dataset(image_set="train", args=args)
    dataset_val = build_dataset(image_set="val", args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )
    data_loader_val = DataLoader(
        dataset_val,
        1,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )
    base_ds = get_coco_api_from_dataset(dataset_val)

    ### LEARNING RATE ###
    if args.onecyclelr:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            steps_per_epoch=len(data_loader_train),
            epochs=args.epochs,
            pct_start=0.2,
        )
    elif args.multi_step_lr:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=args.lr_drop_list
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    ### CHECKPOINT ###
    if args.eval_all:
        step = 100
        model = model_without_ddp
        for checkpoint_path in sorted(glob.glob(f"{model_dir}/checkpoint*.pth")):
            fprint(f"Evaluating {checkpoint_path}...", color="yellow")

            if "checkpoint.pth" in checkpoint_path:
                # only evaluate numbered checkpoints
                continue

            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if run is not None:
                epoch = checkpoint["epoch"]
                run.log({"step": step * epoch})
                run.log({"epoch": epoch})

            model.load_state_dict(checkpoint["model"])
            try:
                stats = evaluate(model, criterion, postprocessors, data_loader_val, base_ds, device, wo_class_error=wo_class_error, args=args)
                main_stats = {
                    "loss": stats["loss"],
                    "loss_param": stats["loss_param"],
                    "class_error": stats["class_error"]
                }
                fprint(main_stats)
                if run is not None:
                    run.log(main_stats)
            except Exception as e:
                fprint(f"Error when evaluating {checkpoint_path}", e=e)
                continue

            try:
                ap = evaluate_ap(model, data_loader_val, device, postprocessors=postprocessors, run=run, args=args, checkpoint_path=checkpoint_path)
                fprint(ap)
            except Exception as e:
                fprint(f"Error when computing average precision for {checkpoint_path}", e=e)
                continue

        torch.cuda.empty_cache()
        return

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location="cpu")
        model_without_ddp.detr.load_state_dict(checkpoint["model"])

    if os.path.exists(model_dir / "checkpoint.pth"):
        args.resume = f"{model_dir}/checkpoint.pth"

    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume,
                map_location="cpu",
                check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")

        model_without_ddp.load_state_dict(checkpoint["model"])
        if args.use_ema:
            if "ema_model" in checkpoint:
                ema_m.module.load_state_dict(
                    utils.clean_state_dict(checkpoint["ema_model"])
                )
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)

        if (
            not args.eval
            and "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1

    if (not args.resume) and args.pretrain_model_path:
        checkpoint = torch.load(args.pretrain_model_path, map_location="cpu")["model"]

        from collections import OrderedDict

        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        logger.info(f"Ignore keys: {json.dumps(ignorelist, indent=2)}")
        _tmp_st = OrderedDict(
            {
                k: v
                for k, v in utils.clean_state_dict(checkpoint).items()
                if check_keep(k, _ignorekeywordlist)
            }
        
        )

        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        logger.info(str(_load_output))

        if args.use_ema:
            if "ema_model" in checkpoint:
                ema_m.module.load_state_dict(
                    utils.clean_state_dict(checkpoint["ema_model"])
                )
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)

    if args.eval:
        # Only evaluating the model
        os.environ["EVAL_FLAG"] = "TRUE"

        test_stats = evaluate(
            model,
            criterion,
            postprocessors,
            data_loader_val,
            base_ds,
            device,
            wo_class_error=wo_class_error,
            args=args,
        )
        log_stats = {**{f"test_{k}": v for k, v in test_stats.items()}}
        if model_dir and utils.is_main_process():
            with (model_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        _ = evaluate_ap(
            model,
            data_loader_val,
            device,
            postprocessors=postprocessors,
            run=run,
            args=args
        )

        return

    print("Start training")
    start_time = time.time()

    # holds the epoch number for which the loss was minimal: in our case, easier to use wandb to make sense of loss
    best_map_holder = BestMetricHolder(use_ema=args.use_ema)
    print(f"Start epoch {args.start_epoch}")

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            args.clip_max_norm,
            wo_class_error=wo_class_error,
            lr_scheduler=lr_scheduler,
            args=args,
            logger=(logger if args.save_log else None),
            ema_m=ema_m,
            run=run,
        )

        if not args.onecyclelr:
            lr_scheduler.step()

        if model_dir:
            checkpoint_paths = [model_dir / "checkpoint.pth"]
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(model_dir / f"checkpoint{epoch:04}.pth")

            for checkpoint_path in checkpoint_paths:
                weights = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "args": args,
                }
                if args.use_ema:
                    weights.update({
                        "ema_model": ema_m.module.state_dict(),
                    })
                utils.save_on_master(weights, checkpoint_path)

        # eval
        _ = evaluate_ap(model, data_loader_val, device, postprocessors=postprocessors, run=run, args=args)
        # logger.info(_)

        test_stats = evaluate(
            model,
            criterion,
            postprocessors,
            data_loader_val,
            base_ds,
            device,
            wo_class_error=wo_class_error,
            args=args,
            logger=(logger if args.save_log else None),
        )

        # new_test_stats = {
        #     f"test_{k}": v
        #     for k, v in test_stats.items()
        #     if not any(str(d) in k for d in range(5))
        # }

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
        }
        # if run:
        #     run.log({
        #         "train_loss": log_stats["train_loss"],
        #         "val_loss": log_stats["val_loss"],
        #     })
        #     # run.log(log_stats)

        if args.use_ema:
            ema_test_stats, ema_coco_evaluator = evaluate(
                ema_m.module,
                criterion,
                postprocessors,
                data_loader_val,
                base_ds,
                device,
                wo_class_error=wo_class_error,
                args=args,
                logger=(logger if args.save_log else None),
            )
            log_stats.update({f"ema_test_{k}": v for k, v in ema_test_stats.items()})
            map_ema = ema_test_stats["coco_eval_bbox"][0]
            _isbest = best_map_holder.update(map_ema, epoch, is_ema=True)
            if _isbest:
                checkpoint_path = model_dir / "checkpoint_best_ema.pth"
                utils.save_on_master(
                    {
                        "model": ema_m.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args,
                    },
                    checkpoint_path,
                )
        log_stats.update(best_map_holder.summary())

        ep_paras = {"epoch": epoch, "n_parameters": n_parameters}
        log_stats.update(ep_paras)
        try:
            log_stats.update({"now_time": str(datetime.datetime.now())})
        except:
            pass

        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        log_stats["epoch_time"] = epoch_time_str

        if model_dir and utils.is_main_process():
            with info_path.open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

    # remove the copied files.
    copyfilelist = vars(args).get("copyfilelist")
    if copyfilelist and args.local_rank == 0:
        from datasets.data_util import remove

        for filename in copyfilelist:
            print(f"Removing: {filename}")
            remove(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DETR-like training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import re
import sys
from typing import Iterable
import torch
import numpy as np

from evaluation.generate_preds import output_class, scale_positions as pred_scale_positions
from evaluation.generate_gt import process_gt, scale_positions as gt_scale_positions
from evaluation.evaluate import get_prim_score, ap
from util.primitives import PRIMITIVES
from util.utils import slprint, to_device
from util.box_ops import arc_cxcywh2_to_xy3, box_cxcywh_to_xyxy
import util.misc as utils


def log_stat(stat_name):
    if any(str(d) in stat_name for d in range(5)):
        return False
    if "giou" in stat_name:
        return False
    return True


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    wo_class_error=False,
    lr_scheduler=None,
    args=None,
    logger=None,
    ema_m=None,
    run=None,
):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    if not wo_class_error:
        metric_logger.add_meter(
            "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
        )

    _cnt = 0
    if run is not None:
        run.log({"epoch": epoch})

    for samples, targets in metric_logger.log_every(
        data_loader, print_freq=50, header=f"\nEpoch: [{epoch}/{args.epochs}]", logger=logger
    ):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)

            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict

            losses = sum(
                loss_dict[k] * weight_dict[k]
                for k in loss_dict.keys()
                if k in weight_dict
            )

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)
        new_loss_scaled = {
            k: v for k, v in loss_dict_reduced_scaled.items() if log_stat(k)
        }
        new_loss_unscaled = {
            k: v for k, v in loss_dict_reduced_unscaled.items() if log_stat(k)
        }

        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()
        # metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(loss=loss_value, **new_loss_scaled, **new_loss_unscaled)

        if run is not None:
            run.log({"step": _cnt})
            run.log({"loss": loss_value})
            run.log({"class_error": loss_dict_reduced["class_error"]})
            run.log(new_loss_scaled)

        if "class_error" in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced["class_error"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!" * 5)
                break

    if getattr(criterion, "loss_weight_decay", False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, "tuning_matching", False):
        criterion.tuning_matching(epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {
        k: meter.global_avg
        for k, meter in metric_logger.meters.items()
        if meter.count > 0
    }
    if getattr(criterion, "loss_weight_decay", False):
        resstat.update({f"weight_{k}": v for k, v in criterion.weight_dict.items()})
    return resstat


@torch.no_grad()
def evaluate_ap(model, data_loader, device, postprocessors, run=None, args=None, checkpoint_path=None):
    model.eval()
    n_gts = {"line": 0, "circle": 0, "arc": 0}
    tps = {"line": [], "circle": [], "arc": []}
    fps = {"line": [], "circle": [], "arc": []}
    scores = {"line": [], "circle": [], "arc": []}
    thres = {"line": 2, "circle": 2, "arc": 6}

    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            # preds = model(samples, targets if args.use_dn else None)
            preds = model(samples)

        orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        output = postprocessors["param"](preds, orig_sizes, to_xyxy=True)[0]
        out_scores = output["scores"]
        select_mask = out_scores > 0.4

        pred_dict = {
            "parameters": output["parameters"][select_mask],
            "size": orig_sizes,
            "labels": output["labels"][select_mask],
            "scores": out_scores[select_mask],
        }

        for prim_k, prim_type in enumerate(["line", "circle", "arc"]):
            prim_preds, prim_scores = output_class(pred_dict, prim_type)

            for i, target in enumerate(targets):
                prim_gt = target[f"{prim_type}s"]
                target[f"{prim_type}s"] = arc_cxcywh2_to_xy3(prim_gt) if prim_type == "arc" else box_cxcywh_to_xyxy(prim_gt)
                target = {k: v.cpu().numpy() for k, v in target.items()}
                prim_gt = process_gt(target, prim_type)
                target[f"{prim_type}s"] = pred_scale_positions(prim_gt.copy(), (128, 128),(1, 1))

                prim_pred = {
                    f"{prim_type}s": pred_scale_positions(prim_preds.copy(), (128, 128), target["orig_size"]),
                    f"{prim_type}_scores": prim_scores,
                }

                res, tp, fp, score = get_prim_score(prim_type, target, prim_pred, prim_k, thres[prim_type])

                scores[prim_type].append(score)
                tps[prim_type].append(tp)
                fps[prim_type].append(fp)
                n_gts[prim_type] += res[f"n_gt_{prim_type}"]

    avg_prec = {}
    for prim_type in ["line", "circle", "arc"]:
        all_tp = np.concatenate(tps[prim_type])
        all_fp = np.concatenate(fps[prim_type])
        sorted_indices = np.argsort(-np.concatenate(scores[prim_type]))
        # nb_gt = n_gts[prim_type] if n_gts[prim_type] > 0 else [1.0] if len(all_tp) == 0 else [0.0]
        all_tp = np.cumsum(all_tp[sorted_indices]) / n_gts[prim_type]
        all_fp = np.cumsum(all_fp[sorted_indices]) / n_gts[prim_type]
        avg_prec[f"AP_{prim_type} (thres {thres[prim_type]})"] = ap(all_tp, all_fp)

    if run is not None:
        run.log(avg_prec)
    return avg_prec


@torch.no_grad()
def evaluate(
    model,
    criterion,
    postprocessors,
    data_loader,
    base_ds,
    device,
    wo_class_error=False,
    args=None,
    logger=None,
):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter(
            "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
        )
    iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors.keys())
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print(f"useCats: {useCats} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    _cnt = 0
    output_state_dict = {}  # for debug only
    for samples, targets in metric_logger.log_every(
        data_loader, 50, "Test:", logger=logger
    ):
        samples = samples.to(device)

        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)
            # outputs = model(samples)

            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        new_loss_scaled = {
            k: v for k, v in loss_dict_reduced_scaled.items() if log_stat(k)
        }
        new_loss_unscaled = {
            k: v for k, v in loss_dict_reduced_unscaled.items() if log_stat(k)
        }

        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()
        # metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(loss=loss_value, **new_loss_scaled, **new_loss_unscaled)
        # metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        # metric_logger.update(loss=loss_value, **new_loss_scaled, **new_loss_unscaled)
        # metric_logger.update(
        #     loss=sum(loss_dict_reduced_scaled.values()),
        #     **loss_dict_reduced_scaled,
        #     **loss_dict_reduced_unscaled,
        # )
        if "class_error" in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced["class_error"])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors["param"](outputs, orig_target_sizes)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if "segm" in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors["segm"](
                results, outputs, orig_target_sizes, target_sizes
            )
        # res = {
        #     target["image_id"].item(): output
        #     for target, output in zip(targets, results)
        # }

        if args.save_results:
            # res_score = outputs['res_score']
            # res_label = outputs['res_label']
            # res_bbox = outputs['res_bbox']
            # res_idx = outputs['res_idx']

            for i, (tgt, res, outbbox) in enumerate(
                zip(targets, results, outputs["pred_boxes"])
            ):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt["boxes"]
                gt_label = tgt["labels"]
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)

                # img_h, img_w = tgt['orig_size'].unbind()
                # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
                # _res_bbox = res['boxes'] / scale_fct
                _res_bbox = outbbox
                _res_prob = res["scores"]
                _res_label = res["labels"]
                res_info = torch.cat(
                    (_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1
                )
                # import ipdb;ipdb.set_trace()

                if "gt_info" not in output_state_dict:
                    output_state_dict["gt_info"] = []
                output_state_dict["gt_info"].append(gt_info.cpu())

                if "res_info" not in output_state_dict:
                    output_state_dict["res_info"] = []
                output_state_dict["res_info"].append(res_info.cpu())

            # # for debug only
            # import random
            # if random.random() > 0.7:
            #     print("Now let's break")
            #     break

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!" * 5)
                break

    if args.save_results:
        import os.path as osp

        # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
        # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
        savepath = osp.join(args.output_dir, f"results-{utils.get_rank()}.pkl")
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    stats = {
        k: meter.global_avg
        for k, meter in metric_logger.meters.items()
        if meter.count > 0
    }

    return stats

import os
from pathlib import Path
import numpy as np
import glob
import pandas as pd
import argparse

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from util import ROOT_DIR

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    type=str,
    default="main_model",
)
parser.add_argument("--epoch", type=str, default="")
parser.add_argument("--data_folder_name", type=str, default="eida_dataset")
parser.add_argument(
    "--line_thresholds",
    nargs=3,
    type=int,
    default=[0.25, 0.5, 0.75, 1, 2, 3, 4, 5],
    help="List of integer thresholds for lines",
)
parser.add_argument(
    "--circle_thresholds",
    nargs=3,
    type=int,
    default=[0.25, 0.5, 0.75, 1, 2, 3, 4, 5],
    help="List of integer thresholds for circles",
)
parser.add_argument(
    "--arc_thresholds",
    nargs=3,
    type=int,
    default=[2, 3, 4, 5, 6, 7, 8, 9],
    help="List of integer thresholds for arcs",
)


def get_l2_distance(pred_circles, gt_circles):
    diff = ((pred_circles[:, None, :, None] - gt_circles[:, None]) ** 2).sum(-1)
    diff = np.minimum(
        (np.sqrt(diff[:, :, 0, 0] + diff[:, :, 1, 1]) / 2),
        (np.sqrt(diff[:, :, 0, 1] + diff[:, :, 1, 0]) / 2),
    )
    return diff


def get_4_cardinal_pts_circle(pred_circles):
    pred_circles_centers = (pred_circles[:, 0, :] + pred_circles[:, 1, :]) / 2
    pred_circles_min_x = np.minimum(pred_circles[:, 0, 0], pred_circles[:, 1, 0])
    pred_circles_max_x = np.maximum(pred_circles[:, 0, 0], pred_circles[:, 1, 0])
    pred_circles_min_y = np.minimum(pred_circles[:, 0, 1], pred_circles[:, 1, 1])
    pred_circles_max_y = np.maximum(pred_circles[:, 0, 1], pred_circles[:, 1, 1])
    pt1 = pred_circles_centers.copy()
    pt1[:, 0] = pred_circles_min_x
    pt2 = pred_circles_centers.copy()
    pt2[:, 0] = pred_circles_max_x
    pt3 = pred_circles_centers.copy()
    pt3[:, 1] = pred_circles_min_y
    pt4 = pred_circles_centers.copy()
    pt4[:, 1] = pred_circles_max_y
    return np.hstack([pt1, pt2, pt3, pt4])


def get_l2_distance_circles(pred_circles, gt_circles):

    cardinal_pts_pred = get_4_cardinal_pts_circle(pred_circles)
    cardinal_pts_gt = get_4_cardinal_pts_circle(gt_circles)

    diff = ((cardinal_pts_pred[:, None, :] - cardinal_pts_gt) ** 2).sum(-1) / 4
    diff = np.sqrt(diff)
    return diff


def get_l2_distance_arcs(pred_arcs, gt_arcs):
    pred_arcs_start_end = pred_arcs[:, :2, :]
    gt_arcs_start_end = gt_arcs[:, :2, :]
    diff_start_end = (
        (pred_arcs_start_end[:, None, :2, None] - gt_arcs_start_end[:, None]) ** 2
    ).sum(-1)
    diff_mid = ((pred_arcs[:, None, 2, :] - gt_arcs[:, 2, :]) ** 2).sum(-1)
    diff = np.minimum(
        np.sqrt(
            (diff_start_end[:, :, 0, 0] + diff_start_end[:, :, 1, 1] + diff_mid) / 3
        ),
        np.sqrt(
            (diff_start_end[:, :, 0, 1] + diff_start_end[:, :, 1, 0] + diff_mid) / 3
        ),
    )
    return diff


def get_precision_recall_fscore(tp, fp, fn):
    if tp == 0:
        if fn == 0:
            return 0, 0, 1
        else:
            return 0, 0, 0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fscore = 2 * precision * recall / (precision + recall)
    return precision, recall, fscore


def msTPFP(line_pred, line_gt, threshold, get_fn=False, primitive_k=0):
    if (len(line_gt) > 0) and (len(line_pred) > 0):
        if primitive_k == 0:
            diff = get_l2_distance(line_pred, line_gt)
        elif primitive_k == 1:
            diff = get_l2_distance_circles(line_pred, line_gt)
        else:
            diff = get_l2_distance_arcs(line_pred, line_gt)

        choice = np.argmin(diff, 1)
        dist = np.min(diff, 1)
        hit = np.zeros(len(line_gt), bool)
        tp = np.zeros(len(line_pred), float)
        fp = np.zeros(len(line_pred), float)

        for i in range(len(line_pred)):
            if dist[i] < threshold and not hit[choice[i]]:
                hit[choice[i]] = True
                tp[i] = 1
            else:
                fp[i] = 1
        fn = 1 - hit
    elif len(line_gt) == 0:
        tp = np.zeros(len(line_pred), float)
        fp = np.ones(len(line_pred))
        fn = np.zeros(len(line_gt), float)
    else:
        tp = np.zeros(len(line_pred), float)
        fp = np.zeros(len(line_pred), float)
        fn = np.ones(len(line_gt), float)
    if get_fn:
        return tp.sum(), fp.sum(), fn.sum()

    return tp, fp


def ap(tp, fp):
    recall = tp
    precision = tp / np.maximum(tp + fp, 1e-9)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    i = np.where(recall[1:] != recall[:-1])[0]
    return np.sum((recall[i + 1] - recall[i]) * precision[i + 1])


def line_score(
    preds_folder,
    gt_path_format,
    line_threshold=5,
    circle_threshold=5,
    arc_threshold=5,
    per_image=True,
):
    gts = sorted(glob.glob(gt_path_format))
    print(len(gts))

    n_gt_lines, n_gt_circles, n_gt_arcs = 0, 0, 0
    lines_tp, lines_fp, lines_scores = [], [], []
    circles_tp, circles_fp, circles_scores, results = [], [], [], []
    arcs_tp, arcs_fp, arcs_scores = [], [], []

    for gt_path in gts:

        im_name = os.path.basename(gt_path).split(".")[0]
        pred_path = Path(preds_folder) / (im_name + ".npz")
        with np.load(pred_path) as fpred:
            pred_line, line_score = fpred["lines"], fpred["line_scores"]
            pred_circle, circle_score = fpred["circles"], fpred["circle_scores"]
            pred_arc, arc_score = fpred["arcs"], fpred["arc_scores"]

        with np.load(gt_path) as fgt:
            try:
                gt_line = fgt["lines"][:, :, :2]
            except IndexError:
                gt_line = []
            try:
                gt_circle = fgt["circles"][:, :, :2]
            except IndexError:
                gt_circle = []
            try:
                gt_arc = fgt["arcs"][:, :, :2]
            except IndexError:
                gt_arc = []
        n_gt_lines += len(gt_line)
        n_gt_circles += len(gt_circle)
        n_gt_arcs += len(gt_arc)
        tp, fp = msTPFP(
            pred_line,
            gt_line,
            line_threshold,
            primitive_k=0,
        )
        lines_tp.append(tp)
        lines_fp.append(fp)
        lines_scores.append(line_score)

        indices = np.argsort(-line_score)
        n_gt_line = len(gt_line)
        if n_gt_line > 0:
            ap_res_line = ap(
                np.cumsum(tp[indices]) / n_gt_line,
                np.cumsum(fp[indices]) / n_gt_line,
            )
        else:
            ap_res_line = 1 if len(fp) == 0 else 0

        tp, fp = msTPFP(
            pred_circle,
            gt_circle,
            circle_threshold,
            primitive_k=1,
        )
        circles_tp.append(tp)
        circles_fp.append(fp)
        circles_scores.append(circle_score)

        indices = np.argsort(-circle_score)
        n_gt_circle = len(gt_circle)
        if n_gt_circle > 0:
            ap_res_circle = ap(
                np.cumsum(tp[indices]) / n_gt_circle,
                np.cumsum(fp[indices]) / n_gt_circle,
            )
        else:
            ap_res_circle = 1 if len(fp) == 0 else 0

        tp, fp = msTPFP(pred_arc, gt_arc, arc_threshold, primitive_k=2)
        arcs_tp.append(tp)
        arcs_fp.append(fp)
        arcs_scores.append(arc_score)

        indices_arcs = np.argsort(-arc_score)
        n_gt_arc = len(gt_arc)
        if n_gt_arc > 0:
            ap_res = ap(
                np.cumsum(tp[indices_arcs]) / n_gt_arc,
                np.cumsum(fp[indices_arcs]) / n_gt_arc,
            )
        else:
            ap_res = 1 if len(fp) == 0 else 0
        results.append(
            {
                "image": im_name,
                "AP_line": ap_res_line,
                "AP_circle": ap_res_circle,
                "AP_arc": ap_res,
                "n_gt_line": n_gt_line,
                "n_gt_circle": n_gt_circle,
                "n_gt_arc": n_gt_arc,
            }
        )

    tps = [lines_tp, circles_tp, arcs_tp]
    fps = [lines_fp, circles_fp, arcs_fp]
    scores = [lines_scores, circles_scores, arcs_scores]
    n_gts = [n_gt_lines, n_gt_circles, n_gt_arcs]
    aps = []
    for all_tp, all_fp, all_scores, n_gt in zip(tps, fps, scores, n_gts):
        all_tp = np.concatenate(all_tp)
        all_fp = np.concatenate(all_fp)
        all_index = np.argsort(-np.concatenate(all_scores))
        all_tp = np.cumsum(all_tp[all_index]) / n_gt
        all_fp = np.cumsum(all_fp[all_index]) / n_gt
        aps.append(ap(all_tp, all_fp))
    results.append(
        {
            "image": "all",
            "AP_line": aps[0],
            "AP_circle": aps[1],
            "AP_arc": aps[2],
            "n_gt_line": n_gt_lines,
            "n_gt_circle": n_gt_circles,
            "n_gt_arc": n_gt_arcs,
        }
    )
    return aps, results


if __name__ == "__main__":
    args = parser.parse_args()
    data_folder = args.data_folder_name
    exp_folder = ROOT_DIR / f"logs/{args.model_name}/"
    GT_path = ROOT_DIR / f"data/{data_folder}/valid_labels/*.npz"

    pred_folder = str(exp_folder / f"npz_preds{args.epoch}")

    eval_folder = exp_folder / f"evaluation{args.epoch}"
    os.makedirs(eval_folder, exist_ok=True)

    line_thresholds, circle_thresholds = args.line_thresholds, args.circle_thresholds
    arc_thresholds = args.arc_thresholds
    for line_threshold, circle_threshold, arc_threshold in zip(
        line_thresholds, circle_thresholds, arc_thresholds
    ):

        aps, results = line_score(
            pred_folder,
            str(GT_path),
            line_threshold=line_threshold,
            circle_threshold=circle_threshold,
            arc_threshold=arc_threshold,
        )

        df_name = f"results_{line_threshold}_{circle_threshold}.csv"

        df = pd.DataFrame(results)
        df["line_threshold"] = line_threshold
        df["circle_threshold"] = circle_threshold
        df["arc_threshold"] = arc_threshold
        df["AP_line"] = np.round(df["AP_line"], 3)
        df["AP_circle"] = np.round(df["AP_circle"], 3)
        df["AP_arc"] = np.round(df["AP_arc"], 3)
        df.to_csv(eval_folder / df_name)
        print(
            f"line_threshold: {line_threshold}, AP_line: {aps[0]}, circle_threshold: {circle_threshold}, ap_circle: {aps[1]}, ap_arc: {aps[2]}, arc_threshold: {arc_threshold}"
        )

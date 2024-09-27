import os
from pathlib import Path
import numpy as np
import glob
import pandas as pd
import argparse

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from util import ROOT_DIR
from util.logger import fprint

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


def get_l2_distance(pred_lines, gt_lines):
    diff = ((pred_lines[:, None, :, None] - gt_lines[:, None]) ** 2).sum(-1)
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


def msTPFP(prim_pred, prim_gt, threshold, get_fn=False, primitive_k=0):
    gt_size = prim_gt.size if type(prim_gt) == np.ndarray else len(prim_gt)
    pred_size = prim_pred.size if type(prim_pred) == np.ndarray else len(prim_pred)

    if (gt_size > 0) and (pred_size > 0):
        if primitive_k == 0:
            diff = get_l2_distance(prim_pred, prim_gt)
        elif primitive_k == 1:
            diff = get_l2_distance_circles(prim_pred, prim_gt)
        else:
            diff = get_l2_distance_arcs(prim_pred, prim_gt)

        choice = np.argmin(diff, 1)
        dist = np.min(diff, 1)
        hit = np.zeros(len(prim_gt), bool)
        tp = np.zeros(len(prim_pred), float)
        fp = np.zeros(len(prim_pred), float)

        for i in range(len(prim_pred)):
            if dist[i] < threshold and not hit[choice[i]]:
                hit[choice[i]] = True
                tp[i] = 1
            else:
                fp[i] = 1
        fn = 1 - hit
    elif gt_size == 0:
        tp = np.zeros(len(prim_pred), float)
        fp = np.ones(len(prim_pred))
        fn = np.zeros(len(prim_gt), float)
    else:
        tp = np.zeros(len(prim_pred), float)
        fp = np.zeros(len(prim_pred), float)
        fn = np.ones(len(prim_gt), float)
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


def get_prim_score(prim_type, fgt, fpred, prim_k=0, threshold=5):
    pred, score = fpred[f"{prim_type}s"], fpred[f"{prim_type}_scores"]

    try:
        gt = fgt[f"{prim_type}s"][:, :, :2]
    except (IndexError, TypeError):
        gt = []

    tp, fp = msTPFP(pred, gt, threshold, primitive_k=prim_k)

    indices = np.argsort(-score)
    n_gt = len(gt)
    if n_gt > 0:
        ap_res = ap(
            np.cumsum(tp[indices]) / n_gt,
            np.cumsum(fp[indices]) / n_gt,
        )
    else:
        ap_res = 1 if len(fp) == 0 else 0

    result = {
        f"AP_{prim_type}": ap_res,
        f"n_gt_{prim_type}": n_gt,
    }

    return result, tp, fp, score


def get_scores(
    preds_folder: Path,
    gt_folder: Path,
    l_threshold=5,
    c_threshold=5,
    a_threshold=5,
    per_image=True,
):
    gts = sorted(glob.glob(f"{gt_folder}/valid_labels/*.npz"))
    print(f"Number of diagrams: {len(gts)}")

    n_gts = {"line": 0, "circle": 0, "arc": 0}
    tps = {"line": [], "circle": [], "arc": []}
    fps = {"line": [], "circle": [], "arc": []}
    scores = {"line": [], "circle": [], "arc": []}
    thres = {"line": l_threshold, "circle": c_threshold, "arc": a_threshold}

    results = []

    for gt_path in gts:
        base_name = os.path.basename(gt_path).split(".")[0]

        result = {
            "image": base_name,
        }

        with np.load(preds_folder / f"{base_name}.npz") as fpred, np.load(gt_path) as fgt:
            for prim_k, prim_type in enumerate(["line", "circle", "arc"]):
                res, tp, fp, score = get_prim_score(prim_type, fgt, fpred, prim_k, thres[prim_type])
                scores[prim_type].append(score)
                tps[prim_type].append(tp)
                fps[prim_type].append(fp)
                n_gts[prim_type] += res[f"n_gt_{prim_type}"]
                result.update(res)

        results.append(result)

    aps = []
    for all_tp, all_fp, all_scores, n_gt in zip(list(tps.values()), list(fps.values()), list(scores.values()), list(n_gts.values())):
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
            "n_gt_line": n_gts["line"],
            "n_gt_circle": n_gts["circle"],
            "n_gt_arc": n_gts["arc"],
        }
    )
    fprint({
        f"AP_line (thres {thres['line']})": aps[0],
        f"AP_circle (thres {thres['circle']})": aps[1],
        f"AP_arc (thres {thres['arc']})": aps[2],
    })
    return aps, results


if __name__ == "__main__":
    args = parser.parse_args()

    model_name = args.model_name

    model_folder = ROOT_DIR / "logs" / args.model_name
    gt_folder = ROOT_DIR / "data" / args.data_folder_name
    pred_folder = gt_folder / f"npz_preds_{model_name}{args.epoch}"
    eval_folder = gt_folder / f"evaluation_{model_name}{args.epoch}"

    os.makedirs(eval_folder, exist_ok=True)

    for l_threshold, c_threshold, a_threshold in zip(
        args.line_thresholds, args.circle_thresholds, args.arc_thresholds
    ):

        aps, results = get_scores(
            pred_folder,
            gt_folder,
            l_threshold=l_threshold,
            c_threshold=c_threshold,
            a_threshold=a_threshold,
        )

        df = pd.DataFrame(results)
        df["line_threshold"] = l_threshold
        df["circle_threshold"] = c_threshold
        df["arc_threshold"] = a_threshold
        df["AP_line"] = np.round(df["AP_line"], 3)
        df["AP_circle"] = np.round(df["AP_circle"], 3)
        df["AP_arc"] = np.round(df["AP_arc"], 3)
        df.to_csv(eval_folder / f"results_{l_threshold}-{a_threshold}.csv")

        print(f"""
        🪈 LINE	Threshold: {l_threshold:.2f}	Average precision:	{aps[0]}
        🪩 CIRCLE	Threshold: {c_threshold:.2f}	Average precision:	{aps[1]}
        󠁼🌈 ARC	Threshold: {a_threshold:.2f}	Average precision:	{aps[2]}
        """)

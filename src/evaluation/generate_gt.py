import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import json
from glob import glob
import argparse
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, default="../data/eida_dataset")


def scale_positions(lines, heatmap_scale=(128, 128), im_shape=None):
    if len(lines) == 0:
        return []
    fx, fy = heatmap_scale[0] / im_shape[0], heatmap_scale[1] / im_shape[1]

    lines[:, :, 0] = np.clip(lines[:, :, 0] * fx, 0, heatmap_scale[0] - 1e-4)
    lines[:, :, 1] = np.clip(lines[:, :, 1] * fy, 0, heatmap_scale[1] - 1e-4)

    return lines


def find_circle_center(p1, p2, p3):
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
    if abs(det) < 1.0e-10:
        return (None, None)

    cx = (bc * (p2[1] - p3[1]) - cd * (p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
    return np.array([cx, cy])


def get_angles_from_arc_points(p0, p_mid, p1):
    arc_center = find_circle_center(p0, p_mid, p1)
    start_angle = np.arctan2(p0[1] - arc_center[1], p0[0] - arc_center[0])
    end_angle = np.arctan2(p1[1] - arc_center[1], p1[0] - arc_center[0])
    mid_angle = np.arctan2(p_mid[1] - arc_center[1], p_mid[0] - arc_center[0])
    return start_angle, mid_angle, end_angle, arc_center


def get_bbox_from_center_radii(center, radii):
    center = np.array(center)
    radii = np.array(radii)
    xs = np.column_stack((center[:, 0] - radii, center[:, 0] + radii))
    ys = np.column_stack((center[:, 1] - radii, center[:, 1] + radii))
    bbox = np.dstack((xs, ys))

    return bbox


def main(data_root, exist_ok=True):
    batch = "valid"
    output_dir = data_root / f"{batch}_labels"
    os.makedirs(output_dir, exist_ok=exist_ok)
    anno_file = os.path.join(data_root, f"{batch}.json")
    im_rescale = (512, 512)
    heatmap_scale = (128, 128)
    with open(anno_file, "r") as f:
        dataset = json.load(f)

    # num_lines = 0
    # num_circles = 0
    # num_arcs = 0
    # avg_num_primitves = 0

    for data in dataset:
        print(data["filename"])
        lines = np.array(data["lines"]).reshape(-1, 2, 2)
        if len(data["arcs"]) == 0:
            arcs = []
        else:
            arcs = np.array(data["arcs"]).reshape(-1, 3, 2)  # p0, p1, pmid

        if len(data["circle_centers"]) == 0:
            circles = []
        else:
            circles = get_bbox_from_center_radii(
                data["circle_centers"], data["circle_radii"]
            )

        im_path = str((data_root / "images") / data["filename"])

        image = Image.open(im_path).convert("RGB")
        im_shape = image.size
        # print(im_shape)

        pos_l = scale_positions(lines.copy(), heatmap_scale, im_shape)
        pos_c = scale_positions(circles.copy(), heatmap_scale, im_shape)
        pos_a = scale_positions(arcs.copy(), heatmap_scale, im_shape)

        image = image.resize(im_rescale)
        prefix = data["filename"].split(".")[0]
        # num_lines += len(lines)
        # num_circles += len(circles)
        # num_arcs += len(arcs)
        # avg_num_primitves += len(lines) + len(circles) + len(arcs)

        # continue
        np.savez_compressed(
            output_dir / f"{prefix}.npz",
            aspect_ratio=image.size[0] / image.size[1],
            lines=pos_l,
            circles=pos_c,
            arcs=pos_a,
        )
        image.save(str(output_dir / f"{prefix}.png"))
        print(f"Saved {prefix}.png")

    # print(f"Total lines: {num_lines}")
    # print(f"Total circles: {num_circles}")
    # print(f"Total arcs: {num_arcs}")
    # print(f"Total primitives: {avg_num_primitves}")
    # print(f"Average primitives per image: {avg_num_primitves/len(dataset):.2}")


if __name__ == "__main__":
    args = parser.parse_args()
    args.data_root = Path(args.data_root)
    main(args.data_root)

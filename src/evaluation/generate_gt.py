import os
import numpy as np
import json
import argparse
from PIL import Image

from ..util import DATA_DIR
from ..util.primitives import PRIM_INFO

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, default="eida_dataset", help="root directory of the data")


def scale_positions(prims, heatmap_scale=(128, 128), im_shape=None):
    if len(prims) == 0:
        return []
    fx, fy = heatmap_scale[0] / im_shape[0], heatmap_scale[1] / im_shape[1]

    prims[:, :, 0] = np.clip(prims[:, :, 0] * fx, 0, heatmap_scale[0] - 1e-4)
    prims[:, :, 1] = np.clip(prims[:, :, 1] * fy, 0, heatmap_scale[1] - 1e-4)

    return prims


def get_bbox_from_center_radii(center, radii):
    center = np.array(center)
    radii = np.array(radii)
    xs = np.column_stack((center[:, 0] - radii, center[:, 0] + radii))
    ys = np.column_stack((center[:, 1] - radii, center[:, 1] + radii))
    bbox = np.dstack((xs, ys))

    return bbox


def process_gt(data, prim_type="line"):
    p_info = PRIM_INFO[prim_type]
    if "circle_centers" in data and prim_type == "circle":
        if len(data["circle_centers"]) == 0:
            return []
        return get_bbox_from_center_radii(data["circle_centers"], data["circle_radii"])

    p_gt = data[f"{prim_type}s"]
    if len(p_gt) == 0:
        return []
    return np.array(data[f"{prim_type}s"]).reshape(p_info["prim_shape"])


def main(data_root, exist_ok=True):
    data_dir = DATA_DIR / data_root
    output_dir = data_dir / "valid_labels"
    os.makedirs(output_dir, exist_ok=exist_ok)
    anno_file = os.path.join(data_dir, "valid.json")

    im_rescale = (512, 512)
    heatmap_scale = (128, 128)

    with open(anno_file, "r") as f:
        dataset = json.load(f)

    # num_lines = 0
    # num_circles = 0
    # num_arcs = 0
    # avg_num_primitves = 0

    for data in dataset:
        filename = data["filename"]
        print(filename)

        image = Image.open(f"{data_dir}/images/{filename}").convert("RGB")
        im_shape = image.size

        # lines = np.array(data["lines"]).reshape(-1, 2, 2)
        # arc = np.array(data["arcs"]).reshape(-1, 3, 2) if len(data["arcs"]) > 0 else []
        # if len(data["circle_centers"]) == 0:
        #    circles = []
        # else:
        #     circles = get_bbox_from_center_radii(
        #         data["circle_centers"], data["circle_radii"]
        #     )

        lines = process_gt(data, "line")
        circles = process_gt(data, "circle")
        arcs = process_gt(data, "arc")

        pos_l = scale_positions(lines.copy(), heatmap_scale, im_shape)
        pos_c = scale_positions(circles.copy(), heatmap_scale, im_shape)
        pos_a = scale_positions(arcs.copy(), heatmap_scale, im_shape)

        image = image.resize(im_rescale)
        prefix = filename.split(".")[0]
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
    args.data_root = args.data_root
    main(args.data_root)

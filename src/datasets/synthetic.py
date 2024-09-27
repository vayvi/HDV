#!/usr/bin/env python3
"""

Examples:
    python synthetic.py synthetic_raw synthetic_processed

Arguments:
                    Source directory that stores synthetic data
                    Temporary output directory

Options:
   -h --help             Show this screen.
"""

import json
import cv2
import os
import numpy as np
from tqdm import tqdm
import argparse

# MARKER : model to process manually annotated data into coco format / add arcs

parser = argparse.ArgumentParser(description="Process raw synthetic data into coco format annotations")
def get_args():
    parser.add_argument("--src", type=str, default="data/synthetic_raw", help="Source directory that stores synthetic data")
    parser.add_argument("--tgt", type=str, default="data/synthetic_processed", help="Output directory for processed synthetic data")
    args = parser.parse_args()
    return args

def main(args):
    src_dir = args.src
    tgt_dir = args.tgt

    image_id = 0
    anno_id = 0

    for batch in ["val", "train"]:
        anno_file = os.path.join(src_dir, f"{batch}.json")

        with open(anno_file, "r") as f:
            dataset = json.load(f)

        def handle(data, image_id, anno_id): # FIXME: remove the anno global variable T_T (when you can test it)
            im = cv2.imread(os.path.join(src_dir, "images", data["filename"]))
            anno["images"].append(
                {
                    "file_name": data["filename"],
                    "height": im.shape[0],
                    "width": im.shape[1],
                    "id": image_id,
                }
            )
            line_set, circle_set = None, None
            if len(data["lines"]) > 0:
                line_set = np.array(data["lines"], dtype=np.float64).reshape(-1, 2, 2)
            # FIXME: attempting initial representation of circles: (x, y, r)
            if len(data["circle_centers"]) > 0:
                circle_set = (data["circle_centers"], data["circle_radii"])


            line_set, circle_set = xyxy_to_xyhw(line_set, circle_set)

            os.makedirs(os.path.join(tgt_dir, batch), exist_ok=True)
            image_path = os.path.join(tgt_dir, batch, data["filename"])
            cv2.imwrite(f"{image_path}", im[::, ::])

            for line in line_set:
                info = {}
                info["id"] = anno_id
                anno_id += 1
                info["image_id"] = image_id
                info["category_id"] = 0
                info["line"] = line
                info["area"] = 1
                anno["annotations"].append(info)
            for circle in circle_set:
                info = {}
                info["id"] = anno_id
                anno_id += 1
                info["image_id"] = image_id
                info["category_id"] = 1
                info["circle"] = circle
                info["area"] = 1  # FIXME: area
                anno["annotations"].append(info)
            image_id += 1
            return anno_id

        anno = {}
        anno["images"] = []
        anno["annotations"] = []
        anno["categories"] = [
            {"supercategory": "line", "id": "0", "name": "line"},
            {"supercategory": "circle", "id": "1", "name": "circle"},
        ]

        for img_dict in tqdm(dataset):
            anno_id = handle(img_dict, image_id, anno_id)
            image_id += 1
            anno_path = os.path.join(tgt_dir, "annotations", f"primitives_{batch}.json")
        os.makedirs(os.path.join(tgt_dir, "annotations"), exist_ok=True)

        with open(anno_path, "w") as outfile:
            json.dump(anno, outfile)


def xyxy_to_xyhw(lines=None, circles=None):  # TODO: maybe optimize with numpy? 
    # change the format from x,y,x,y to x,y,dx, dy
    # order: top point > bottom point
    #        if same y coordinate, right point > left point
    new_lines_pairs = []
    if lines is not None:
        for line in lines:  # [ #lines, 2, 2 ]
            p1 = line[0]  # xy
            p2 = line[1]  # xy
            if p1[0] < p2[0]:
                new_lines_pairs.append([p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1]])
            elif p1[0] > p2[0]:
                new_lines_pairs.append([p2[0], p2[1], p1[0] - p2[0], p1[1] - p2[1]])
            else:
                if p1[1] < p2[1]:
                    new_lines_pairs.append([p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1]])
                else:
                    new_lines_pairs.append([p2[0], p2[1], p1[0] - p2[0], p1[1] - p2[1]])
    new_circles = []
    if circles is not None:
        centers, radii = circles

        for center, radius in zip(centers, radii):
            new_circles.append([center[0], center[1], radius, radius])

    return new_lines_pairs, new_circles


if __name__ == "__main__":
    args = get_args()
    main(args)

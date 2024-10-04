import json
import os
from pathlib import Path
import argparse
from xml.dom import minidom

import numpy as np
from PIL import Image

from datasets.coco import CocoDetection
from inference import save_pred_as_img
from .util import DATA_DIR
from .util.primitives import (
    get_angles_from_arc_points,
    get_arc_param_with_tr,
    get_arc_param_from_inkscape,
    read_paths_with_transforms,
    get_radius,
    get_circles_from_ellipses,
    BadPath,
    PRIM_INFO
)
from .util.logger import SLogger

parser = argparse.ArgumentParser()
parser.add_argument("--data_set", default="eida_dataset", type=str, help="Name of the folder containing SVG and images to be used as ground truth")
parser.add_argument("--sanity_check", action="store_true", help="Create images out of converted annotation to verify correctness")
parser.add_argument("--train_portion", default=0.8, help="Portion of the dataset used for training (rest is used for validation): between 0 and 1")


"""
output.json should have the following format:
{
    "images": [
        {
            "file_name": "<filename>.<ext>",
            "height": <px>,
            "width": <px>,
            "id": <unique_img_id>
        },
        { ... }
    ],
    "annotations": [
        {
            "id": <unique_anno_id>,
            "category_id": <prim_id>, # 0 for lines, 1 for circles, 2 for arcs
            "image_id": <unique_img_id>,
            "parameters": 
                # for lines (x,y then relative coordinates): 
                [[x1, y1], [dx, d2]]
                # for circles:
                [center_x, center_y, radius, radius]
                # for arcs (start, end, mid_point):
                [x1, y1, x2, y2, x3, y3]
        },
        { ... }
    ]
}  
"""
output = { "images": {}, "annotations": {} }

def draw_arc(param, img, width_ratio, color="firebrick"):
    p0 = np.array([param[0], param[1]])
    p1 = np.array([param[2], param[3]])
    p_mid = np.array([param[4], param[5]])
    img.point(p0, fill=color)
    img.point(p_mid, fill=color)
    img.point(p1, fill=color)
    (
        start_angle,
        mid_angle,
        end_angle,
        center,
    ) = get_angles_from_arc_points(p0, p_mid, p1)
    radius = np.linalg.norm(p0 - center)
    shape = [
        (center[0] - radius, center[1] - radius),
        (center[0] + radius, center[1] + radius),
    ]
    to_deg = lambda x: (x * 180 / np.pi)

    img.arc(
        shape,
        start=to_deg(start_angle),
        end=to_deg(mid_angle),
        fill=color,
        width=int(2 * width_ratio),
    )
    img.arc(
        shape,
        start=to_deg(mid_angle),
        end=to_deg(end_angle),
        fill=color,
        width=int(2 * width_ratio),
    )

def draw_line(param, img, width_ratio, color="green"):
    line = [(param[0][0], param[0][1]), (param[1][0], param[1][1])]
    img.line(line, fill=color, width=int(2 * width_ratio))

def draw_circle(param, img, width_ratio, color="royalblue"):
    cx, cy, radius, _ = param
    img.ellipse(
        [
            cx - radius,
            cy - radius,
            cx + radius,
            cy + radius,
        ],
        outline=color,
        width=int(2 * width_ratio),
    )


def svg_to_params(svg_path):
    doc = minidom.parse(str(svg_path))
    img_file = os.path.basename(doc.getElementsByTagName("image")[0].getAttribute("xlink:href"))
    circle_r, circle_c, arc_coords = [], [], []

    path_and_transforms = []
    for path in doc.getElementsByTagName("path"):
        transform_string = path.getAttribute("transform")
        try:
            arc_coords.append(get_arc_param_from_inkscape(path))
        except ValueError as e:
            if path.getAttribute("d"):
                path_and_transforms.append(
                    (path.getAttribute("d"), transform_string)
                )
            elif path.getAttribute("inkscape:original-d"):
                path_and_transforms.append(
                    (path.getAttribute("inkscape:original-d"), transform_string)
                )
            else:
                raise BadPath(f"Invalid arc path {path}")

    doc_circles = doc.getElementsByTagName("circle")
    doc_ellipses = doc.getElementsByTagName("ellipse")

    if len(doc_circles) > 0:
        circle_r = np.array([get_radius(c) for c in doc_circles])
        circle_c = np.array(
            [
                [float(c.getAttribute("cx")), float(c.getAttribute("cy"))]
                for c in doc_circles
            ]
        )
    if len(doc_ellipses) > 0:
        ellipses, (circle_c, circle_r) = get_circles_from_ellipses(doc_ellipses, (circle_c, circle_r))

    doc.unlink()

    lines_coords, (all_arcs, arc_transforms) = read_paths_with_transforms(path_and_transforms)

    for arc_path, arc_transform in zip(all_arcs, arc_transforms):
        arc_coords.append(get_arc_param_with_tr(arc_path, arc_transform))

    lines, arcs, circles = [], [], []

    for circle_pos, circle_radius in zip(circle_c, circle_r):
        circles.append([circle_pos[0], circle_pos[1], circle_radius, circle_radius])

    for line_coords in np.array(lines_coords):
        delta_x = float(line_coords[2]) - float(line_coords[0])
        delta_y = float(line_coords[3]) - float(line_coords[1])
        lines.append(
            [
                [float(line_coords[0]), float(line_coords[1])],
                [delta_x, delta_y]
            ]
        )
    for arc_coord in arc_coords:
        arcs.append(
            [
                float(arc_coord[0][0]),
                float(arc_coord[0][1]),
                float(arc_coord[1][0]),
                float(arc_coord[1][1]),
                float(arc_coord[2][0]),
                float(arc_coord[2][1]),
            ]
        )

    return {
        "line": lines,
        "arc": arcs,
        "circle": circles,
    }, img_file

def save_dataset(set_name, annotations):
    # save training data annotations
    with open(out_folder / "annotations" / f"primitives_{set_name}.json", "w") as f:
        json.dump({
            "images": list(annotations["images"].values()),
            "annotations": list(annotations["annotations"].values()),
        }, f, indent=4)

    dataset_img = out_folder / set_name
    logger.info(f"""
    SAVED {set_name.upper()} ANNOTATIONS:
    Annotations: {len(annotations['annotations'])}
    Images: {len(annotations['images'])}
    Files in {dataset_img}: {len(list(dataset_img.glob('*')))}""")

    if args.sanity_check:
        dataset = CocoDetection(
            out_folder / data_set,
            out_folder / "annotations" / f"primitives_{data_set}.json",
            transforms=None,
            args=None,
        )
        # vslzr = COCOVisualizer()
        for _img, _anno in dataset:
            im_name = Path(annotations["images"][int(_anno["image_id"][0])]["file_name"]).stem
            save_pred_as_img(im_name, _img, _anno, svg_folder)
            # id_to_prim = {value['id']: key for key, value in PRIM_INFO.items()}
            # preds = {
            #     'parameters': _anno['parameters'],
            #     'image_id': _anno['image_id'],
            #     'size': _anno["size"],
            #     'labels': [id_to_prim[value.item()] for value in _anno["labels"]],
            # }
            # vslzr.visualize(
            #     _img,
            #     preds,
            #     primitives_to_show=list(PRIM_INFO.keys()),
            #     show_boxes=False,
            #     show_text=False,
            #     show_image=True,
            #     savedir=out_folder / data_set,
            #     img_name=f"{im_name}.jpg",
            # )

    # reset output annotations
    return {"images": {}, "annotations": {}}, "val"

if __name__ == "__main__":
    args = parser.parse_args()

    data_folder = DATA_DIR / Path(args.data_set)
    train_portion = args.train_portion
    assert train_portion > 0
    assert train_portion <= 1

    svg_folder = data_folder / "svgs"
    img_folder = data_folder / "images"
    out_folder = data_folder / "groundtruth"

    # Create finetuning data folder structure
    Path(out_folder).mkdir(parents=True, exist_ok=True)
    for folder in ["annotations", "train", "val"]:
        Path(out_folder / folder).mkdir(parents=True, exist_ok=True)

    logger = SLogger(
        name="convert_svg",
        log_file=svg_folder / f"logs_svg_to_train.txt",
    )

    for folder in [data_folder, svg_folder, img_folder]:
        if not folder.exists():
            raise FileNotFoundError(f"Folder {folder} does not exist")

    svg_files = list(svg_folder.glob("*.svg"))
    if len(svg_files) == 0:
        raise FileNotFoundError(f"No SVG files found in {svg_folder}")

    total = len(svg_files)
    logger.info(f"Found {total} SVG files in {svg_folder}")
    data_set = "train"

    for nb, file in enumerate(svg_files):
        if nb / total > train_portion and data_set == "train":
            output, data_set = save_dataset(data_set, output)

        try:
            params, img_name = svg_to_params(file)
        except Exception as e:
            logger.error(f"Error processing {file}", e)
            continue

        try:
            img = Image.open(img_folder / img_name).convert("RGB")
        except Exception as e:
            logger.error(f"Error with image {img_folder / img_name}", e)
            continue

        img.save(out_folder / data_set / img_name)
        h, w = img.size

        output["images"][nb] = {
            "file_name": str(out_folder / data_set / img_name),
            "height": h,
            "width": w,
            "id": nb,
        }

        prim_id = 0
        for prim_type in params:
            for _, p in enumerate(params[prim_type]):
                output["annotations"][f"{nb}_{prim_id}"] = {
                    "id": f"{nb}_{prim_id}",
                    "category_id": PRIM_INFO[prim_type]["id"],
                    "image_id": nb,
                    f"{prim_type}": p,
                }

                prim_id += 1

        # if args.sanity_check:
        #     width_ratio = min(max((h + w) // 2 / 600, 2), 5)
        #     img1 = ImageDraw.Draw(img)
        #
        #     for arc in params["arc"]:
        #         draw_arc(arc, img1, width_ratio)
        #
        #     for line in params["line"]:
        #         draw_line(line, img1, width_ratio)
        #
        #     for circle in params["circle"]:
        #         draw_circle(circle, img1, width_ratio)
        #
        #     img.save(svg_folder / f"{img_name.split('.')[0]}_out.png")

    save_dataset(data_set, output)

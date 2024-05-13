import json
import os
from pathlib import Path
import argparse
from xml.dom import minidom

import numpy as np
from PIL import Image, ImageDraw

from util.primitives import (
    get_angles_from_arc_points,
    get_arc_param_with_tr,
    get_arc_param_from_inkscape,
    read_paths_with_transforms,
    get_radius,
    BadPath, PRIM_INFO
)
from util.logger import SLogger

parser = argparse.ArgumentParser()
parser.add_argument("--svg_folder", type=str, help="Folder containing SVG files to process")
parser.add_argument("--img_folder", type=str, help="Folder containing images files associated with SVG files")
parser.add_argument("--sanity_check", action="store_true", help="Create images out of converted annotation to verify correctness")

data_sets = {
    "train": 0.8,
    "val": 0.2,
}

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
                # for lines: 
                [[x1, y1], [x2, y2]]
                # for circles:
                [center_x, center_y, radius, radius]
                # for arcs (start, end, mid point):
                [x1, y1, x2, y2, x3, y3]
        },
        { ... }
    ]
}  
"""
output = {
    "images": [],
    "annotations": [],
}

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
    line = [(param[0], param[1]), (param[2], param[3])]
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


def svg_to_params(svg_path, ellipse_to_circle_ratio_threshold=5 * 1e-2):
    doc = minidom.parse(str(svg_path))
    img_file = os.path.basename(doc.getElementsByTagName("image")[0].getAttribute("xlink:href"))
    ellipses, circle_r, circle_centers, arc_params = {}, [], [], []

    path_strings_and_transforms = []
    for path in doc.getElementsByTagName("path"):
        transform_string = path.getAttribute("transform")
        try:
            arc_params.append(get_arc_param_from_inkscape(path))
        except ValueError as e:
            if path.getAttribute("d"):
                path_strings_and_transforms.append(
                    (path.getAttribute("d"), transform_string)
                )
            elif path.getAttribute("inkscape:original-d"):
                path_strings_and_transforms.append(
                    (path.getAttribute("inkscape:original-d"), transform_string)
                )
            else:
                raise BadPath(f"Invalid arc path {path}")

    doc_circles, doc_ellipses = doc.getElementsByTagName(
        "circle"
    ), doc.getElementsByTagName("ellipse")

    if len(doc_circles) > 0:
        circle_r = np.array([get_radius(circle) for circle in doc_circles])
        circle_centers = np.array(
            [
                [float(circle.getAttribute("cx")), float(circle.getAttribute("cy"))]
                for circle in doc_circles
            ]
        )
    if len(doc_ellipses) > 0:
        ellipse_centers = np.array(
            [
                [float(ellipse.getAttribute("cx")), float(ellipse.getAttribute("cy"))]
                for ellipse in doc_ellipses
            ]
        )
        ellipse_r = np.array(
            [
                [float(ellipse.getAttribute("rx")), float(ellipse.getAttribute("ry"))]
                for ellipse in doc_ellipses
            ]
        )

        mask = (
            np.abs((ellipse_r[:, 0] / (ellipse_r[:, 1] + 1e-8)) - 1)
            < ellipse_to_circle_ratio_threshold
        )
        if len(circle_centers):
            circle_centers = np.vstack([circle_centers, ellipse_centers[mask]])
            circle_r = np.concatenate([circle_r, np.mean(ellipse_r[mask], axis=1)])
        else:
            circle_centers = ellipse_centers[mask]
            circle_r = np.mean(ellipse_r[mask], axis=1)
        ellipse_centers, ellipse_r = ellipse_centers[~mask], ellipse_r[~mask]
        if len(ellipse_centers) > 0:
            ellipses = {"ellipse_centers": ellipse_centers, "ellipse_radii": ellipse_r}
            print("############")
            print(f"svg {svg_path} has ellipses.")

    doc.unlink()

    lines_c, (all_arcs, arc_transforms) = read_paths_with_transforms(
        path_strings_and_transforms
    )

    for arc_path, arc_transform in zip(all_arcs, arc_transforms):
        arc_params.append(get_arc_param_with_tr(arc_path, arc_transform))

    lines, arc_coords = [], []
    circles = []
    for circle_pos, circle_radius in zip(circle_centers, circle_r):
        circles.append([circle_pos[0], circle_pos[1], circle_radius, circle_radius])

    for line_coords in np.array(lines_c):
        lines.append(
            [
                float(line_coords[0]),
                float(line_coords[1]),
                float(line_coords[2]),
                float(line_coords[3]),
            ]
        )
    for arc_coord in arc_params:
        arc_coords.append(
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
        "arc": arc_coords,
        "circle": circles,
    }, img_file

if __name__ == "__main__":
    args = parser.parse_args()

    svg_folder = Path(args.svg_folder)
    img_folder = Path(args.img_folder)

    logger = SLogger(
        name="convert_svg",
        log_file=svg_folder / f"logs_svg_to_train.txt",
    )

    if not svg_folder.exists():
        raise FileNotFoundError(f"Folder {svg_folder} does not exist")

    svg_files = list(svg_folder.glob("*.svg"))
    if len(svg_files) == 0:
        raise FileNotFoundError(f"No SVG files found in {svg_folder}")

    train_json = svg_folder / "train.json"
    val_json = svg_folder / "val.json"

    for nb, file in enumerate(svg_files):
        # TODO check that the file contains an image that exists
        try:
            params, img_name = svg_to_params(file)
        except Exception as e:
            logger.error(f"Error processing {file}", e)
            continue

        img = Image.open(img_folder / img_name).convert("RGB")
        h, w = img.size

        output["images"].append(
            {
                "file_name": str(img_folder / img_name),
                "height": h,
                "width": w,
                "id": nb,
            }
        )
        for prim_type in params:
            for prim_id, p in enumerate(params[prim_type]):
                output["annotations"].append(
                    {
                        "id": f"{nb}_{prim_id}",
                        "category_id": PRIM_INFO[prim_type]["id"],
                        "image_id": nb,
                        "parameters": p,
                    }
                )

        if args.sanity_check:
            width_ratio = min(max((h + w) // 2 / 600, 2), 5)
            img1 = ImageDraw.Draw(img)

            for arc_param in params["arc"]:
                draw_arc(arc_param, img1, width_ratio)

            for line in params["line"]:
                draw_line(line, img1, width_ratio)

            for circle in params["circle"]:
                draw_circle(circle, img1, width_ratio)

            img.save(svg_folder / f"{img_name}_out.png")

    with open(train_json, "w") as f:
        json.dump(output, f, indent=4)

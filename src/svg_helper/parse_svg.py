import sys

import numpy as np
from xml.dom import minidom
import glob
import json
import os
from pathlib import Path
import argparse
import re
from PIL import Image, ImageDraw

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from util.primitives import (
    get_angles_from_arc_points,
    get_arc_param_with_tr,
    get_arc_param_from_inkscape,
    read_paths_with_transforms,
    BadPath,
    get_radius
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_folder", type=str, default="eida_dataset", help="input folder"
)
parser.add_argument("--svg_folder", type=str, default="svgs")
parser.add_argument("--sanity_check", action="store_true")
parser.add_argument("--image_file_extension", type=str, default="jpg")
parser.add_argument("--exist_ok", action="store_true")


def get_gt_from_svg(annotation_path, ellipse_to_circle_ratio_threshold=5 * 1e-2):
    doc = minidom.parse(str(annotation_path))
    svg_params = doc.getElementsByTagName("svg")[0]
    width, height = svg_params.getAttribute("width"), svg_params.getAttribute("height")
    image_link = doc.getElementsByTagName("image")[0].getAttribute("xlink:href")
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
            print(f"svg {annotation_path} has ellipses.")

    doc.unlink()

    lines, (all_arcs, arc_transforms) = read_paths_with_transforms(
        path_strings_and_transforms
    )

    for arc_path, arc_transform in zip(all_arcs, arc_transforms):
        arc_params.append(get_arc_param_with_tr(arc_path, arc_transform))

    return {
        "line_coords": np.array(lines),
        "circle_pos": circle_centers,
        "circle_radius": circle_r,
        "arc_coords": arc_params,
        "ellipses": ellipses,
        "width": float(width),
        "height": float(height),
    }


def get_annotation(table):
    centers, circle_radii, lines, arc_coords = [], [], [], []
    for circle_pos, circle_radius in zip(table["circle_pos"], table["circle_radius"]):
        center = [circle_pos[0], circle_pos[1]]
        centers.append(center)
        circle_radii.append(circle_radius)

    for line_coords in table["line_coords"]:
        lines.append(
            [
                float(line_coords[0]),
                float(line_coords[1]),
                float(line_coords[2]),
                float(line_coords[3]),
            ]
        )
    for arc_coord in table["arc_coords"]:
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
        "circle_centers": centers,
        "circle_radii": circle_radii,
        "lines": lines,
        "arcs": arc_coords,
        "width": table["width"],
        "height": table["height"],
    }



if __name__ == "__main__":
    args = parser.parse_args()
    annotations = []
    parent_folder_path = Path("data") / args.input_folder
    print(parent_folder_path / args.svg_folder)
    if args.sanity_check:
        os.makedirs(parent_folder_path / "sanity_check_resized", exist_ok=args.exist_ok)

    for svg_path in glob.glob(str(parent_folder_path / args.svg_folder) + "/*corr.svg"):
        print("current svg path", svg_path)
        name = os.path.basename(svg_path).split("_corr.svg")[0]
        try:
            table = get_gt_from_svg(svg_path)
        except BadPath as e:
            print(e, f" #### Skipping {svg_path}")
            continue
        annotation = get_annotation(table)
        annotation["filename"] = f"{name}.{args.image_file_extension}"
        annotations.append(annotation)

        im_file_path = (
            parent_folder_path / "images" / f"{name}.{args.image_file_extension}"
        )

        if args.sanity_check:
            img = Image.open(im_file_path).convert("RGB")
            img_size = (min(img.size) + max(img.size)) // 2
            width_ratio = min(max(img_size / 600, 2), 5)
            img1 = ImageDraw.Draw(img)

            for arc_param in annotation["arcs"]:
                p0 = np.array([arc_param[0], arc_param[1]])
                p1 = np.array([arc_param[2], arc_param[3]])
                p_mid = np.array([arc_param[4], arc_param[5]])
                img1.point(p0, fill="firebrick")
                img1.point(p_mid, fill="firebrick")
                img1.point(p1, fill="firebrick")
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
                to_rad = lambda x: x * np.pi / 180

                def get_tuple_from_two_points(p0, p1):
                    return (p0[0], p0[1], p1[0], p1[1])

                start_angle, end_angle, mid_angle = (
                    to_deg(start_angle),
                    to_deg(end_angle),
                    to_deg(mid_angle),
                )

                img1.arc(
                    shape,
                    start=start_angle,
                    end=mid_angle,
                    fill="firebrick",
                    width=int(2 * width_ratio),
                )
                img1.arc(
                    shape,
                    start=mid_angle,
                    end=end_angle,
                    fill="firebrick",
                    width=int(2 * width_ratio),
                )

            for line in annotation["lines"]:
                line = [(line[0], line[1]), (line[2], line[3])]
                img1.line(line, fill="green", width=int(2 * width_ratio))

            for circle_center, circle_radius in zip(
                annotation["circle_centers"], annotation["circle_radii"]
            ):
                img1.ellipse(
                    [
                        circle_center[0] - circle_radius,
                        circle_center[1] - circle_radius,
                        circle_center[0] + circle_radius,
                        circle_center[1] + circle_radius,
                    ],
                    outline="royalblue",
                    width=int(2 * width_ratio),
                )

            img.save(parent_folder_path / f"sanity_check_resized/{name}.png")
    with open(parent_folder_path / "valid.json", "w") as json_file:
        json.dump(annotations, json_file)
        print(f"Successfully created {parent_folder_path / 'valid.json'}")

import numpy as np
from svg.path import parse_path
from svg.path.path import Line, Move, Arc
from xml.dom import minidom
import glob
import json
import os
from pathlib import Path
import argparse
import re
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_folder", type=str, default="eida_dataset", help="input folder"
)
parser.add_argument("--svg_folder", type=str, default="svgs")
parser.add_argument("--sanity_check", action="store_true")
parser.add_argument("--image_file_extension", type=str, default="jpg")
parser.add_argument("--exist_ok", action="store_true")

def parse_matrix_string(matrix_string):
    assert not matrix_string.startswith("translate")
    if matrix_string.startswith("matrix"):
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", matrix_string)
        numbers = [float(num) for num in numbers]
        matrix = np.eye(3)
        matrix[0, :3] = numbers[0], numbers[2], numbers[4]
        matrix[1, :3] = numbers[1], numbers[3], numbers[5]

        return matrix

    elif matrix_string.startswith("scale"):
        numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", matrix_string)
        numbers = [float(num) for num in numbers]

        if len(numbers) == 1:
            numbers = numbers * 2
        matrix = np.eye(3)
        matrix[0, 0] = numbers[0]
        matrix[1, 1] = numbers[1]
        return matrix

    elif matrix_string.startswith("rotate"):
        numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", matrix_string)
        numbers = [float(num) for num in numbers]

        assert len(numbers) == 1, "rotation angle should be one number"
        angle_radians = np.radians(numbers[0])
        matrix_2d = np.array(
            [
                [np.cos(angle_radians), -np.sin(angle_radians)],
                [np.sin(angle_radians), np.cos(angle_radians)],
            ]
        )
        matrix = np.eye(3)
        matrix[:2, :2] = matrix_2d
        return matrix
    else:
        raise Exception(f"Unknown transform {matrix_string}")


class BadPath(Exception):
    pass


supported_transforms = ["matrix", "scale"]


def is_mergeable(arc1, arc2):
    connected = arc1.end == arc2.start
    same_center = (arc1.radius == arc2.radius) and (arc1.rotation == arc2.rotation)
    same_orientation = (arc1.sweep == arc2.sweep) and (arc1.arc == arc2.arc)
    return connected and same_center and same_orientation


def merge_arcs(arc_list):
    indices_to_remove = []
    for i, (arc1, arc2) in enumerate(zip(arc_list[:-1], arc_list[1:])):
        if is_mergeable(arc1, arc2):
            arc2.start = arc1.start
            indices_to_remove.append(i)
    return [arc for i, arc in enumerate(arc_list) if i not in indices_to_remove]


def read_paths_with_transforms(path_strings_and_transforms):
    lines, all_arcs, arc_transforms = [], [], []
    for path_string_and_transform in path_strings_and_transforms:
        e, transform = path_string_and_transform
        path = parse_path(e)
        # print(path)
        arcs_in_path = []
        for e in path:
            if isinstance(e, Line):
                x0, y0 = e.start.real, e.start.imag
                x1, y1 = e.end.real, e.end.imag
                if transform != "":
                    transform_matrix = parse_matrix_string(transform)
                    print("line transform matrix", transform_matrix)

                    x0, y0 = (transform_matrix @ np.append(np.array([x0, y0]), 1))[:2]

                    x1, y1 = (transform_matrix @ np.append(np.array([x1, y1]), 1))[:2]
                line = np.array([x0, y0, x1, y1])
                if np.linalg.norm(line) > 1e-5:
                    lines.append(line)
            elif isinstance(e, Arc):
                arcs_in_path.append(e)
        if len(arcs_in_path) > 0:
            all_arcs.append(merge_arcs(arcs_in_path))
            arc_transforms.append(transform)
    return lines, (all_arcs, arc_transforms)


def is_large_arc(rad_angle):
    if rad_angle[0] <= np.pi:
        return not (rad_angle[0] < rad_angle[1] < (np.pi + rad_angle[0]))
    return (rad_angle[0] - np.pi) < rad_angle[1] < rad_angle[0]


def get_arc_param_from_inkscape(arc_path_object):
    arc_cx = float(arc_path_object.getAttribute("sodipodi:cx"))
    arc_cy = float(arc_path_object.getAttribute("sodipodi:cy"))
    arc_rx = float(arc_path_object.getAttribute("sodipodi:rx"))
    arc_ry = float(arc_path_object.getAttribute("sodipodi:ry"))
    arc_start_angle = float(arc_path_object.getAttribute("sodipodi:start"))
    arc_end_angle = float(arc_path_object.getAttribute("sodipodi:end"))

    arc_center = np.array([arc_cx, arc_cy])
    arc_radius = np.array([arc_rx, arc_ry])
    p0 = (
        arc_cx + arc_rx * np.cos(arc_start_angle),
        arc_cy + arc_ry * np.sin(arc_start_angle),
    )
    p1 = (
        arc_cx + arc_rx * np.cos(arc_end_angle),
        arc_cy + arc_ry * np.sin(arc_end_angle),
    )
    if arc_start_angle > arc_end_angle:
        arc_end_angle += 2 * np.pi
    arc_mid_angle = arc_start_angle + (arc_end_angle - arc_start_angle) / 2
    p_mid = (
        arc_cx + arc_rx * np.cos(arc_mid_angle),
        arc_cy + arc_ry * np.sin(arc_mid_angle),
    )
    arc_transform = arc_path_object.getAttribute("transform")
    if arc_transform != "":
        print(arc_transform)
        transform_matrix = parse_matrix_string(arc_transform)
        p0 = (transform_matrix @ np.append(p0, 1))[:2]
        p1 = (transform_matrix @ np.append(p1, 1))[:2]

        # NOTE: check determinant of the matrix to not change the flag of the arc
        p_mid = (transform_matrix @ np.append(p_mid, 1))[:2]
        if np.linalg.det(transform_matrix[:2, :2]) < 0:
            p0, p1 = p1, p0
    return p0, p1, p_mid
    # # TODO: deduce points first because there might be a rotation afterwards
    # arc_center = (transform_matrix @ np.append(arc_center, 1))[:2]
    # arc_radius = (transform_matrix @ np.append(arc_radius, 1))[:2]
    # if np.abs((arc_radius[0] / arc_radius[1]) - 1) > 1e-1:
    #     raise BadPath("arc path with non-circular radius", arc_radius)
    # radius = (arc_radius[0] + arc_radius[1]) / 2

    # p0 = (
    #     arc_center[0] + radius * np.cos(start_angle),
    #     arc_center[1] + radius * np.sin(start_angle),
    # )
    # p1 = (
    #     arc_center[0] + radius * np.cos(end_angle),
    #     arc_center[1] + radius * np.sin(end_angle),
    # )

    # p_mid = (arc_center[0] + radius * np.cos(mid_angle), arc_center[1] + radius * np.sin(mid_angle))
    # p1 = (arc_center[0] + radius * np.cos(end_angle), arc_center[1] + radius * np.sin(end_angle))


def get_arc_param(arc_path, arc_transform):
    to_2pi = lambda x: (x + 2 * np.pi) % (2 * np.pi)
    assert len(arc_path) == 1, f"arc path with more than one arc {arc}"
    arc_path = arc_path[0]
    assert arc_path.sweep, f"arc path with negative sweep {arc_path}"
    assert arc_path.rotation == 0, f"arc path with non-zero rotation {arc_path}"
    if arc_path.arc:
        raise BadPath(f"arc path with large arc {arc_path}")
    # assert not arc_path.arc, f"arc path with large arc {arc_path}"

    p0 = np.array([arc_path.start.real, arc_path.start.imag])
    p1 = np.array([arc_path.end.real, arc_path.end.imag])
    center = np.array([arc_path.center.real, arc_path.center.imag])
    # print(arc_transform)
    if arc_transform.startswith("translate"):
        numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", arc_transform)
        numbers = [float(num) for num in numbers]

        if len(numbers) == 1:
            numbers = numbers.append(0)
        numbers = np.array(numbers)
        print("translating with ", numbers)
        p0 += numbers
        p1 += numbers
        center += numbers

    elif arc_transform != "":
        transform_matrix = parse_matrix_string(arc_transform)

        p0 = (transform_matrix @ np.append(p0, 1))[:2]
        p1 = (transform_matrix @ np.append(p1, 1))[:2]

        if np.linalg.det(transform_matrix[:2, :2]) < 0:
            p0, p1 = p1, p0

        center = (transform_matrix @ np.append(center, 1))[:2]
    if np.abs((np.linalg.norm(p0 - center) / np.linalg.norm(p1 - center)) - 1) > 1e-1:
        raise BadPath("arc path with non-circular radius", arc_path)
    # radius = (np.linalg.norm(p0 - center) + np.linalg.norm(p1 - center)) / 2
    radius = np.linalg.norm(p0 - center)

    start_angle = to_2pi(np.arctan2(p0[1] - center[1], p0[0] - center[0]))
    end_angle = to_2pi(np.arctan2(p1[1] - center[1], p1[0] - center[0]))
    large_arc_flag = is_large_arc([start_angle, end_angle])

    if start_angle > end_angle:
        end_angle += 2 * np.pi
    # if large_arc_flag:
    #     mid_angle = start_angle + (end_angle - start_angle) / 2 + np.pi
    # else:
    mid_angle = start_angle + (end_angle - start_angle) / 2
    mid_angle, end_angle = mid_angle % (2 * np.pi), end_angle % (2 * np.pi)
    p_mid = center + radius * np.array([np.cos(mid_angle), np.sin(mid_angle)])
    # return p0, p1, center, radius, start_angle, end_angle, mid_angle, p_mid, large_arc_flag
    return p0, p1, p_mid


def get_radius(circle):
    try:
        r = float(circle.getAttribute("r"))
    except ValueError as e:
        try:
            rx = float(circle.getAttribute("rx"))
            ry = float(circle.getAttribute("ry"))
            if np.abs(1 - ry / rx) < 1e-2:
                r = (rx + ry) / 2
            else:
                raise BadPath(f"shape is coded as circle with rx,ry={rx} , {ry}")
        except Exception as e:
            print(e)
            raise BadPath(f"Invalid circle")
    return r


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
        arc_params.append(get_arc_param(arc_path, arc_transform))

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

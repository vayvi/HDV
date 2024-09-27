import re

import numpy as np

import matplotlib.pyplot as plt
from svg.path import parse_path
from svg.path.path import Line, Move, Arc


PRIM_INFO = {
    'line': {'id': 0, 'color': 'red', 'line_width': 1, 'indices': slice(0, 4), 'param_shape': (-1, 2, 2), 'prim_shape': (-1, 2, 2), 'x_border_idx': [0, 2], 'y_border_idx': [1, 3]},
    'circle': {'id': 1, 'color': 'green', 'line_width': 1, 'indices': slice(4, 8), 'param_shape': (-1, 3), 'prim_shape': (-1, 2, 2), 'x_border_idx': [0, 2], 'y_border_idx': [1, 3]},
    'arc': {'id': 2, 'color': 'blue', 'line_width': 1, 'indices': slice(8, 14), 'param_shape': (-1, 3, 2), 'prim_shape': (-1, 3, 2), 'x_border_idx': [0, 2, 4], 'y_border_idx': [1, 3, 5]}
}
PRIMITIVES = list(PRIM_INFO.keys())


class BadPath(Exception):
    pass


def get_circles_from_ellipses(ellipses, circles, ellipse_to_circle_ratio_threshold=5 * 1e-2):
    c_centers, c_radii = circles
    e_centers = np.array(
        [
            [float(ellipse.getAttribute("cx")), float(ellipse.getAttribute("cy"))]
            for ellipse in ellipses
        ]
    )
    e_radii = np.array(
        [
            [float(ellipse.getAttribute("rx")), float(ellipse.getAttribute("ry"))]
            for ellipse in ellipses
        ]
    )

    mask = (
        np.abs((e_radii[:, 0] / (e_radii[:, 1] + 1e-8)) - 1)
        < ellipse_to_circle_ratio_threshold
    )
    if len(c_centers):
        c_centers = np.vstack([c_centers, e_centers[mask]])
        c_radii = np.concatenate([c_radii, np.mean(e_radii[mask], axis=1)])
    else:
        c_centers = e_centers[mask]
        c_radii = np.mean(e_radii[mask], axis=1)

    e_centers, e_radii = e_centers[~mask], e_radii[~mask]
    if len(e_centers) > 0:
        ellipses = {"ellipse_centers": e_centers, "ellipse_radii": e_radii}
        print(f"Found ellipses.")

    return ellipses, (c_centers, c_radii)


def get_angles_from_arc_points(p0, p_mid, p1):
    arc_center = find_circle_center(p0, p_mid, p1)
    arc_center = (arc_center[0], arc_center[1]) # NOTE some versions don't have this line
    start_angle = np.arctan2(p0[1] - arc_center[1], p0[0] - arc_center[0])
    end_angle = np.arctan2(p1[1] - arc_center[1], p1[0] - arc_center[0])
    mid_angle = np.arctan2(p_mid[1] - arc_center[1], p_mid[0] - arc_center[0])
    return start_angle, mid_angle, end_angle, arc_center


def get_arc_plot_params(arc):
    start_angle, mid_angle, end_angle, arc_center = get_angles_from_arc_points(
        arc[:2],
        arc[4:],
        arc[2:4],
    )
    # print(start_angle, mid_angle, end_angle)
    diameter = 2 * np.linalg.norm(arc[:2] - arc_center)
    to_deg = lambda x: (x * 180 / np.pi) % 360
    start_angle, mid_angle, end_angle = (
        to_deg(start_angle),
        to_deg(mid_angle),
        to_deg(end_angle),
    )
    # print("angles", start_angle, mid_angle, end_angle)
    return start_angle, mid_angle, end_angle, arc_center, diameter


def box_xyxy_to_cxcyr(x):
    """
    Only valid for circles
    """
    x0, y0, x1, y1 = x.T
    b = np.stack([(x0 + x1) / 2, (y0 + y1) / 2,
                  ((x1 - x0) + (y1 - y0))/4], axis=-1)
    return b


def calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    angle1 = np.arctan2(v1[1], v1[0])
    angle2 = np.arctan2(v2[1], v2[0])
    angle = angle1 - angle2
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return angle


def write_svg_dwg(dwg, lines=None, circles=None, arcs=None, show_image=False, image=None):
    # Add the background image to the drawing
    from matplotlib.patches import Polygon, Circle

    if show_image:
        dpi = 100
        fig = plt.figure(dpi=dpi)
        plt.rcParams["font.size"] = "5"
        ax = plt.gca()
        ax.imshow(image)

    for circle in circles:
        cx, cy = circle[:2]
        radius = circle[-1]
        if show_image:
            circle_plot = Circle(circle[:2], circle[-1], fill=None, color="red", linewidth=1)
            ax.add_patch(circle_plot)
        dwg.add(dwg.circle(center=(str(cx), str(cy)), r=str(radius), fill="none", stroke='blue', stroke_width=3))

    for line in lines:
        try:
            p1x, p1y = line[0]
            p2x, p2y = line[1]
        except:
            p1x, p1y, p2x, p2y = line

        if show_image:
            line_plot = Polygon(line, fill=None, color="red", linewidth=1)
            ax.add_patch(line_plot)

        dwg.add(dwg.path(d="M " + str(p1x) + " " + str(p1y) + " L " + str(p2x) + " " + str(p2y), stroke="green",
                         stroke_width=3, fill="none"))
    for arc in arcs:
        p0, p1, pmid = arc[0], arc[1], arc[2]
        start_angle, mid_angle, end_angle, arc_center = get_angles_from_arc_points(p0, p1, pmid)
        arc_radius = np.linalg.norm(p0 - arc_center)
        large_arc_flag = np.linalg.norm((p0 + p1) / 2 - pmid) > arc_radius
        sweep_flag = calculate_angle(p0, p1, pmid) < 0
        # p1x, p1y = p0
        # p2x, p2y = pmid
        # dwg.add(dwg.path(d="M " + str(p1x) + " " + str(p1y) + " L " + str(p2x) + " " + str(p2y), stroke="red", stroke_width=2, fill="none"))
        # p1x, p1y = pmid
        # p2x, p2y = p1
        # dwg.add(dwg.path(d="M " + str(p1x) + " " + str(p1y) + " L " + str(p2x) + " " + str(p2y), stroke="red", stroke_width=2, fill="none"))
        arc_args = {
            "x0": p0[0],
            "y0": p0[1],
            "xradius": arc_radius,
            "yradius": arc_radius,
            "ellipseRotation": 0,  # has no effect for circles
            "x1": p1[0],
            "y1": p1[1],
            "large_arc_flag": int(large_arc_flag),
            "sweep_flag": int(sweep_flag),  # set sweep-flag to 1 for clockwise arc
        }
        dwg.add(dwg.path(
            d="M %(x0)f,%(y0)f A %(xradius)f,%(yradius)f %(ellipseRotation)f %(large_arc_flag)d,%(sweep_flag)d %(x1)f,%(y1)f"
              % arc_args,
            fill="none",
            stroke="firebrick",
            stroke_width=3,
        ))

    return dwg


def get_arc_param(arc_path, arc_transform=None):
    to_2pi = lambda x: (x + 2 * np.pi) % (2 * np.pi)
    assert len(arc_path) == 1, f"arc path with more than one arc {arc_path}"
    arc_path = arc_path[0]

    assert arc_path.rotation == 0, f"arc path with non-zero rotation {arc_path}"

    p0 = np.array([arc_path.start.real, arc_path.start.imag])
    p1 = np.array([arc_path.end.real, arc_path.end.imag])
    if not arc_path.sweep:
        p0, p1 = p1, p0
    center = np.array([arc_path.center.real, arc_path.center.imag])

    radius = np.linalg.norm(p0 - center)
    start_angle = to_2pi(np.arctan2(p0[1] - center[1], p0[0] - center[0]))
    end_angle = to_2pi(np.arctan2(p1[1] - center[1], p1[0] - center[0]))

    return center, radius, start_angle, end_angle, p0, p1


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
        path_str, transform = path_string_and_transform
        path = parse_path(path_str)
        # print(path)
        arcs_in_path = []
        for p in path:
            if isinstance(p, Line):
                x0, y0 = p.start.real, p.start.imag
                x1, y1 = p.end.real, p.end.imag
                if transform != "":
                    transform_matrix = parse_matrix_string(transform)
                    # print("line transform matrix", transform_matrix)

                    x0, y0 = (transform_matrix @ np.append(np.array([x0, y0]), 1))[:2]
                    x1, y1 = (transform_matrix @ np.append(np.array([x1, y1]), 1))[:2]

                line = np.array([x0, y0, x1, y1])
                if np.linalg.norm(line) > 1e-5:
                    lines.append(line)
            elif isinstance(p, Arc):
                arcs_in_path.append(p)
        if len(arcs_in_path) > 0:
            all_arcs.append(merge_arcs(arcs_in_path))
            arc_transforms.append(transform)
    return lines, (all_arcs, arc_transforms)


def is_large_arc(rad_angle):
    if rad_angle[0] <= np.pi:
        return not (rad_angle[0] < rad_angle[1] < (np.pi + rad_angle[0]))
    return (rad_angle[0] - np.pi) < rad_angle[1] < rad_angle[0]


def get_arc_param_with_tr(arc_path, arc_transform):
    to_2pi = lambda x: (x + 2 * np.pi) % (2 * np.pi)
    assert len(arc_path) == 1, f"arc path with more than one arc {arc_path}"
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


def line_to_xy(x):
    c_x, c_y, w, h = x
    x0, y0 = c_x - w / 2, c_y - h / 2
    x1, y1 = c_x + w / 2, c_y + h / 2
    return x0, y0, x1, y1


def circle_to_xy(x):
    c_x, c_y, w, h = x
    r = (w + h) / 4
    return c_x, c_y, r


def arc_to_xy(x):
    cx, cy, w, h, w_mid, h_mid = x
    x0, y0 = cx - w / 2, cy - h / 2
    x1, y1 = cx + w / 2, cy + h / 2
    x_mid, y_mid = cx + w_mid / 2, cy + h_mid / 2
    return x0, y0, x1, y1, x_mid, y_mid

def find_circle_center(p1, p2, p3):
    """Circle center from 3 points"""
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
    if abs(det) < 1.0e-10:
        return (None, None)

    cx = (bc * (p2[1] - p3[1]) - cd * (p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
    return np.array([cx, cy])


def find_circle_center_arr(p1, p2, p3):
    """Circle center from 3 points"""
    temp = p2[:, 0] ** 2 + p2[:, 1] ** 2
    bc = (p1[:, 0] ** 2 + p1[:, 1] ** 2 - temp) / 2
    cd = (temp - p3[:, 0] ** 2 - p3[:, 1] ** 2) / 2
    det = (p1[:, 0] - p2[:, 0]) * (p2[:, 1] - p3[:, 1]) - (p2[:, 0] - p3[:, 0]) * (
        p1[:, 1] - p2[:, 1]
    )

    # Handle the case where the determinant is close to zero
    mask = np.abs(det) < 1.0e-10
    det[mask] = 1.0  # Prevent division by zero
    bc[mask] = 0.0  # These arcs will have center at (0, 0)
    cd[mask] = 0.0

    cx = (bc * (p2[:, 1] - p3[:, 1]) - cd * (p1[:, 1] - p2[:, 1])) / det
    cy = ((p1[:, 0] - p2[:, 0]) * cd - (p2[:, 0] - p3[:, 0]) * bc) / det
    return np.stack([cx, cy], axis=-1)

def remove_duplicate_circles(circle_coords_list, image_size, circle_scores=None):
    circle_coords = np.array(circle_coords_list).reshape(-1, 3) # (n_circles, 3)
    distances = np.linalg.norm(circle_coords[None, :, :] - circle_coords[:, None, :], axis=-1)
    threshold = (image_size[0] + image_size[1]) / 80
    mask = distances < threshold
    indices_to_remove = np.array([np.sum(row[i+1:]) for i, row in enumerate(mask)])
    indices_to_keep = np.where(indices_to_remove == 0)[0]
    if circle_scores is not None:
        circle_scores = np.array(circle_scores)
        return circle_coords_list[indices_to_keep], circle_scores[indices_to_keep]
    return circle_coords[indices_to_keep]

def remove_duplicate_lines(line_coords_list, image_size, line_scores=None):
    line_coords = np.array(line_coords_list).reshape(-1, 4) # (n_lines, 4)
    permuted_lines = np.hstack((line_coords[:, 2:4],line_coords[:, 0:2]))
    distances = np.minimum(np.linalg.norm(line_coords[None, :, :] - line_coords[:, None, :], axis=-1), np.linalg.norm(line_coords[None, :, :] - permuted_lines[:, None, :], axis=-1))
    threshold = (image_size[0] + image_size[1]) / 80
    mask = distances < threshold
    indices_to_remove = np.array([np.sum(row[i+1:]) for i, row in enumerate(mask)])
    indices_to_keep = np.where(indices_to_remove == 0)[0]
    if line_scores is not None:
        line_scores = np.array(line_scores)
        return line_coords_list[indices_to_keep], line_scores[indices_to_keep]
    return line_coords_list[indices_to_keep]

def remove_small_lines(line_coords_list, image_size, line_scores=None):
    line_coords = np.array(line_coords_list).reshape(-1, 4) # (n_lines, 4)
    lengths = np.linalg.norm(line_coords[:, :2] - line_coords[:, 2:], axis=-1)
    threshold = (image_size[0] + image_size[1]) / 50
    # print(lengths)
    mask = lengths > threshold
    indices_to_keep = np.where(mask)[0]
    if line_scores is not None:
        line_scores = np.array(line_scores)
        return line_coords_list[indices_to_keep], line_scores[indices_to_keep]
    return line_coords_list[indices_to_keep]

def remove_duplicate_arcs(line_coords_list, image_size, line_scores=None):
    line_coords = np.array(line_coords_list).reshape(-1, 6) # (n_lines, 6)
    permuted_lines = np.hstack((line_coords[:, 2:4],line_coords[:, 0:2], line_coords[:, 4:6]))
    distances = np.minimum(np.linalg.norm(line_coords[None, :, :] - line_coords[:, None, :], axis=-1), np.linalg.norm(line_coords[None, :, :] - permuted_lines[:, None, :], axis=-1))
    threshold = (image_size[0] + image_size[1]) / 50
    mask = distances < threshold
    indices_to_remove = np.array([np.sum(row[i+1:]) for i, row in enumerate(mask)])
    indices_to_keep = np.where(indices_to_remove == 0)[0]
    if line_scores is not None:
        line_scores = np.array(line_scores)
        return line_coords_list[indices_to_keep], line_scores[indices_to_keep]
    return line_coords_list[indices_to_keep]

def remove_arcs_on_top_of_circles(arc_coords_list, circle_coords_list, image_size, arc_scores=None):
    arc_coords = np.array(arc_coords_list).reshape(-1, 6) # (n_arcs, 6)
    circle_coords = np.array(circle_coords_list).reshape(-1, 3) # (n_circles, 3)
    arc_centers = find_circle_center_arr(
        arc_coords[:, :2],
        arc_coords[:, 4:],
        arc_coords[:, 2:4],
    )
    radii = np.linalg.norm(arc_coords[:, :2] - arc_centers, axis = 1)
    arc_circle_coords = np.hstack((arc_centers, radii[:, None]))
    distances = np.linalg.norm(circle_coords[None, :, :] - arc_circle_coords[:, None, :], axis=-1)
    threshold = (image_size[0] + image_size[1]) / 80
    # print(distances)
    # print(threshold)
    mask = distances < threshold
    indices_to_remove = np.array([np.sum(row) for i, row in enumerate(mask)])
    indices_to_keep = np.where(indices_to_remove == 0)[0]
    if arc_scores is not None:
        arc_scores = np.array(arc_scores)
        return arc_coords_list[indices_to_keep], arc_scores[indices_to_keep]
    return arc_coords_list[indices_to_keep]

def remove_arcs_on_top_of_lines(arc_coords_list, line_coords_list, image_size, arc_scores=None):
    arc_coords = np.array(arc_coords_list).reshape(-1, 6) # (n_arcs, 6)
    line_coords = np.array(line_coords_list).reshape(-1, 4) # (n_lines, 4)
    line_coords_w_center = np.hstack((line_coords[:, :2], line_coords[:, 2:], (line_coords[:, :2] + line_coords[:, 2:])/2))
    line_coords_w_center_permuted = np.hstack((line_coords[:, 2:], line_coords[:, :2], (line_coords[:, :2] + line_coords[:, 2:])/2))
    distances = np.minimum(np.linalg.norm(line_coords_w_center[None, :, :] - arc_coords[:, None, :], axis=-1), np.linalg.norm(line_coords_w_center_permuted[None, :, :] - arc_coords[:, None, :], axis=-1))
    threshold = (image_size[0] + image_size[1]) / 50
    mask = distances < threshold
    indices_to_remove = np.array([np.sum(row) for i, row in enumerate(mask)])
    indices_to_keep = np.where(indices_to_remove == 0)[0]
    if arc_scores is not None:
        arc_scores = np.array(arc_scores)
        return arc_coords_list[indices_to_keep], arc_scores[indices_to_keep]
    return arc_coords_list[indices_to_keep]

from matplotlib.patches import Arc
def plot_arc(ax, arc, c='r', linewidth=2):
    arc = arc.reshape(-1)
    theta1, theta_mid, theta2, c_xy, diameter = get_arc_plot_params(arc)
    if theta_mid < theta1 and theta_mid > theta2:
        theta1, theta2 = theta2, theta1
    to_rad = lambda x: (x * np.pi / 180) % (2 * np.pi)
    if not is_large_arc([to_rad(theta1), to_rad(theta_mid)]):
        arc_patch_1 = Arc(
            c_xy,
            diameter,
            diameter,
            angle=0.0,
            theta1=theta1,
            theta2=theta_mid,
            fill=None,
            color=c,
            linewidth=linewidth,
        )
    else:
        arc_patch_1 = Arc(
            c_xy,
            diameter,
            diameter,
            angle=0.0,
            theta1=theta_mid,
            theta2=theta1,
            fill=None,
            color=c,
            # color="black",
            linewidth=linewidth,
        )
    ax.add_patch(arc_patch_1)

    if not is_large_arc([to_rad(theta_mid), to_rad(theta2)]):
        arc_patch_2 = Arc(
            c_xy,
            diameter,
            diameter,
            angle=0.0,
            theta1=theta_mid,
            theta2=theta2,
            fill=None,
            color=c,
            linewidth=linewidth,
        )

    else:
        arc_patch_2 = Arc(
            c_xy,
            diameter,
            diameter,
            angle=0.0,
            theta1=theta2,
            theta2=theta_mid,
            fill=None,
            color=c,
            # color="black",
            linewidth=linewidth,
        )
    ax.add_patch(arc_patch_2)

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


def get_arc_param_from_inkscape(arc_path_object):
    arc_cx = float(arc_path_object.getAttribute("sodipodi:cx"))
    arc_cy = float(arc_path_object.getAttribute("sodipodi:cy"))
    arc_rx = float(arc_path_object.getAttribute("sodipodi:rx"))
    arc_ry = float(arc_path_object.getAttribute("sodipodi:ry"))
    arc_start_angle = float(arc_path_object.getAttribute("sodipodi:start"))
    arc_end_angle = float(arc_path_object.getAttribute("sodipodi:end"))

    # for attr in ["cx", "cy", "rx", "ry", "start", "end"]:
    #      arc_path_object.removeAttribute(f"sodipodi:{attr}")

    # arc_center = np.array([arc_cx, arc_cy])
    # arc_radius = np.array([arc_rx, arc_ry])
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
        # print(arc_transform)
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

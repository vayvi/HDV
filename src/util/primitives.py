import numpy as np

import matplotlib.pyplot as plt


def get_angles_from_arc_points(p0, p_mid, p1):
    arc_center = find_circle_center(p0, p_mid, p1)
    arc_center = (arc_center[0], arc_center[1])
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
    assert len(arc_path) == 1, f"arc path with more than one arc {arc}"
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


def is_large_arc(rad_angle):
    if rad_angle[0] <= np.pi:
        return not (rad_angle[0] < rad_angle[1] < (np.pi + rad_angle[0]))
    return (rad_angle[0] - np.pi) < rad_angle[1] < rad_angle[0]

def find_circle_center(p1, p2, p3):
    """Circle center from 3 points"""

    p1_0, p1_1 = p1[0], p1[1]
    p2_0, p2_1 = p2[0], p2[1]
    p3_0, p3_1 = p3[0], p3[1]

    temp = p2_0 * p2_0 + p2_1 * p2_1
    bc = (p1_0 * p1_0 + p1_1 * p1_1 - temp) / 2
    cd = (temp - p3_0 * p3_0 - p3_1 * p3_1) / 2
    det = (p1_0 - p2_0) * (p2_1 - p3_1) - (p2_0 - p3_0) * (p1_1 - p2_1)
    if abs(det) < 1.0e-10:
        return (None, None)

    cx = (bc * (p2_1 - p3_1) - cd * (p1_1 - p2_1)) / det
    cy = ((p1_0 - p2_0) * cd - (p2_0 - p3_0) * bc) / det
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
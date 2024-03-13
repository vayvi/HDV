# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch, os
from torchvision.ops.boxes import box_area


def find_circle_center(p1, p2, p3):
    """Circle center from 3 points"""
    temp = p2[:, 0] ** 2 + p2[:, 1] ** 2
    bc = (p1[:, 0] ** 2 + p1[:, 1] ** 2 - temp) / 2
    cd = (temp - p3[:, 0] ** 2 - p3[:, 1] ** 2) / 2
    det = (p1[:, 0] - p2[:, 0]) * (p2[:, 1] - p3[:, 1]) - (p2[:, 0] - p3[:, 0]) * (
        p1[:, 1] - p2[:, 1]
    )
    # Handle the case where the determinant is close to zero
    mask = torch.abs(det) < 1.0e-10
    det[mask] = 1.0  # Prevent division by zero
    bc[mask] = 0.0  # These arcs will have center at (0, 0)
    cd[mask] = 0.0

    cx = (bc * (p2[:, 1] - p3[:, 1]) - cd * (p1[:, 1] - p2[:, 1])) / det
    cy = ((p1[:, 0] - p2[:, 0]) * cd - (p2[:, 0] - p3[:, 0]) * bc) / det
    return torch.stack([cx, cy], dim=-1)


def get_angles_from_arc_points(p0, p_mid, p1):
    arc_center = find_circle_center(
        p0, p_mid, p1
    ) 
    start_angle = torch.atan2(p0[:, 1] - arc_center[:, 1], p0[:, 0] - arc_center[:, 0])
    end_angle = torch.atan2(p1[:, 1] - arc_center[:, 1], p1[:, 0] - arc_center[:, 0])
    mid_angle = torch.atan2(
        p_mid[:, 1] - arc_center[:, 1], p_mid[:, 0] - arc_center[:, 0]
    )
    to_deg = lambda x: (x * 180 / torch.pi) % 360
    start_angle = to_deg(start_angle)
    end_angle = to_deg(end_angle)
    mid_angle = to_deg(mid_angle)
    return start_angle, mid_angle, end_angle, arc_center


def get_box_from_arcs(arcs):
    x1, y1, x2, y2, xmid, ymid = arcs.t()

    min_x = torch.min(x1, x2)
    min_y = torch.min(y1, y2)
    max_x = torch.max(x1, x2)
    max_y = torch.max(y1, y2)

    start_points = arcs[:, :2]
    mid_points = arcs[:, 4:6]
    end_points = arcs[:, 2:4]

    start_angle, mid_angle, end_angle, arc_center = get_angles_from_arc_points(
        start_points, mid_points, end_points
    )
    radius = torch.norm(start_points - arc_center, dim=1)
    mask_clockwise = (mid_angle - start_angle) % 360 > (end_angle - start_angle) % 360
    start_angle[mask_clockwise], end_angle[mask_clockwise] = (
        end_angle[mask_clockwise],
        start_angle[mask_clockwise],
    )

    cross0 = start_angle > end_angle
    cross90 = (start_angle - 90) % 360 > (end_angle - 90) % 360
    cross180 = (start_angle - 180) % 360 > (end_angle - 180) % 360
    cross270 = (start_angle - 270) % 360 > (end_angle - 270) % 360

    max_x[cross0] = arc_center[cross0, 0] + radius[cross0]
    max_y[cross90] = arc_center[cross90, 1] + radius[cross90]
    min_x[cross180] = arc_center[cross180, 0] - radius[cross180]
    min_y[cross270] = arc_center[cross270, 1] - radius[cross270]

    return torch.stack([min_x, min_y, max_x, max_y], dim=1)



def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcyr(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, ((x1 - x0) + (y1 - y0)) / 4]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh_abs(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, torch.abs(x1 - x0), torch.abs(y1 - y0)]
    return torch.stack(b, dim=-1)


def param_xyxy_to_cxcywh(x):
    line = box_xyxy_to_cxcywh(x[..., :4])
    circle = box_xyxy_to_cxcywh_abs(x[..., 4:8])
    arc = arc_xy3_to_cxcywh2(x[..., 8:14])
    box = box_xyxy_to_cxcywh(x[..., 14:18])
    return torch.cat([line, circle, arc, box], dim=-1)


def param_cxcywh_to_xyxy(x):
    line = box_cxcywh_to_xyxy(x[..., :4])
    circle = box_cxcywh_to_xyxy(x[..., 4:8])
    arc = arc_cxcywh2_to_xy3(x[..., 8:14])
    box = box_cxcywh_to_xyxy(x[..., 14:18])
    return torch.cat([line, circle, arc, box], dim=-1)


def arc_xy3_to_cxcywh2(x):
    x0, y0, x1, y1, x_mid, y_mid = x.unbind(-1)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = x1 - x0, y1 - y0
    w_mid, h_mid = 2 * (x_mid - cx), 2 * (y_mid - cy)
    return torch.stack([cx, cy, w, h, w_mid, h_mid], dim=-1)


def arc_cxcywh2_to_xy3(x):
    cx, cy, w, h, w_mid, h_mid = x.unbind(-1)
    x0, y0 = cx - w / 2, cy - h / 2
    x1, y1 = cx + w / 2, cy + h / 2
    x_mid, y_mid = cx + w_mid / 2, cy + h_mid / 2
    return torch.stack([x0, y0, x1, y1, x_mid, y_mid], dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / (area + 1e-6)


# modified from torchvision to also return the union
def box_iou_pairwise(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N,2]
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N,2]

    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]

    union = area1 + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou_pairwise(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    Input:
        - boxes1, boxes2: N,4
    Output:
        - giou: N, 4
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    assert boxes1.shape == boxes2.shape
    iou, union = box_iou_pairwise(boxes1, boxes2)  # N, 4

    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,2]
    area = wh[:, 0] * wh[:, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


if __name__ == "__main__":
    x = torch.rand(5, 4)
    y = torch.rand(3, 4)
    iou, union = box_iou(x, y)
    import ipdb

    ipdb.set_trace()


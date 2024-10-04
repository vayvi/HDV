# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modules to compute the matching cost and solve the corresponding LSAP.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------


import torch
from torch import nn
from scipy.optimize import linear_sum_assignment
from ...util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, param_cxcywh_to_xyxy


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        focal_alpha=0.25,
        to_xyxy=False,
        min_permut_loss=False,
    ):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.to_xyxy = to_xyxy
        self.min_permut_loss = min_permut_loss
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"

        self.focal_alpha = focal_alpha

    def compute_cost(self, out_param, tgt_param, indices, device):
        if tgt_param.nelement() != 0:
            return torch.cdist(out_param[..., indices], tgt_param[..., indices], p=1)
        else:
            return torch.tensor([], device=device)

    def compute_lines_min_xyxy(self, out_param, tgt_param_lines, device):
        if tgt_param_lines.nelement() != 0:
            cost_lines = torch.min(
                torch.cdist(out_param[..., :4], tgt_param_lines[..., :4], p=1),
                torch.cdist(
                    torch.cat((out_param[..., 2:4], out_param[..., :2]), dim=-1),
                    tgt_param_lines[..., :4],
                    p=1,
                ),
            )
            return cost_lines
        else:
            return torch.tensor([], device=device)

    def compute_lines_min_cxcywh(self, out_param, tgt_param_lines, device):
        if tgt_param_lines.nelement() != 0:
            cost_lines = torch.cdist(out_param[..., :2], tgt_param_lines[..., :2], p=1)
            cost_lines += torch.min(
                torch.cdist(out_param[..., 2:4], tgt_param_lines[..., 2:4], p=1),
                torch.cdist(
                    out_param[..., 2:4] * -1.0,
                    tgt_param_lines[..., 2:4],
                    p=1,
                ),
            )
            return cost_lines
        else:
            return torch.tensor([], device=device)

    def compute_arcs_min_xyxy(self, out_param, tgt_param_arcs, device):
        if tgt_param_arcs.nelement() != 0:
            cost_arcs = torch.min(
                torch.cdist(out_param[..., 8:12], tgt_param_arcs[..., 8:12]),
                torch.cdist(
                    torch.cat((out_param[..., 10:12], out_param[..., 8:10]), dim=-1),
                    tgt_param_arcs[..., 8:12],
                    p=1,
                ),
            )
            cost_arcs += torch.cdist(
                out_param[..., 12:14], tgt_param_arcs[..., 12:14], p=1
            )
            return cost_arcs
        else:
            return torch.tensor([], device=device)

    def compute_arcs_min_cxcywh(self, out_param, tgt_param_arcs, device):
        if tgt_param_arcs.nelement() != 0:
            cost_arcs = torch.cdist(
                out_param[..., 8:10], tgt_param_arcs[..., 8:10], p=1
            )
            cost_arcs += torch.min(
                torch.cdist(out_param[..., 10:12], tgt_param_arcs[..., 10:12], p=1),
                torch.cdist(
                    out_param[..., 10:12] * -1.0,
                    tgt_param_arcs[..., 10:12],
                    p=1,
                ),
            )
            cost_arcs += torch.cdist(
                out_param[..., 12:14], tgt_param_arcs[..., 12:14], p=1
            )
            return cost_arcs
        else:
            return torch.tensor([], device=device)

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = (
            outputs["pred_logits"].flatten(0, 1).sigmoid()
        )  # [batch_size * num_queries, num_classes]
        out_param = outputs["pred_params"].flatten(
            0, 1
        )  # [batch_size * num_queries, 18]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_param = torch.cat([v["parameters"] for v in targets])

        # Compute the classification cost.
        alpha = self.focal_alpha
        gamma = 2.0
        neg_cost_class = (
            (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
        )
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        device = cost_class.device

        if self.to_xyxy:
            out_param_xy = param_cxcywh_to_xyxy(out_param)
            tgt_param_xy = param_cxcywh_to_xyxy(tgt_param)
            tgt_param_lines = tgt_param_xy[tgt_ids == 0]
            tgt_param_circles = tgt_param_xy[tgt_ids == 1]
            tgt_param_arcs = tgt_param_xy[tgt_ids == 2]
            cost_bbox = torch.cdist(
                out_param_xy[..., 14:18], tgt_param_xy[..., 14:18], p=1
            )
            cost_circles = self.compute_cost(
                out_param_xy, tgt_param_circles, slice(4, 8), device
            )

            if self.min_permut_loss:
                cost_lines = self.compute_lines_min_xyxy(
                    out_param_xy, tgt_param_lines, device
                )
                cost_arcs = self.compute_arcs_min_xyxy(
                    out_param_xy, tgt_param_arcs, device
                )
            else:
                cost_lines = self.compute_cost(
                    out_param_xy, tgt_param_lines, slice(0, 4), device
                )
                cost_arcs = self.compute_cost(
                    out_param_xy, tgt_param_arcs, slice(8, 14), device
                )
        else:
            tgt_param_lines = tgt_param[tgt_ids == 0]
            tgt_param_circles = tgt_param[tgt_ids == 1]
            tgt_param_arcs = tgt_param[tgt_ids == 2]
            cost_bbox = torch.cdist(out_param[..., 14:18], tgt_param[..., 14:18], p=1)
            cost_circles = self.compute_cost(
                out_param, tgt_param_circles, slice(4, 8), device
            )
            if self.min_permut_loss:
                cost_lines = self.compute_lines_min_cxcywh(
                    out_param, tgt_param_lines, device
                )
                cost_arcs = self.compute_arcs_min_cxcywh(
                    out_param, tgt_param_arcs, device
                )
            else:
                cost_lines = self.compute_cost(
                    out_param, tgt_param_lines, slice(0, 4), device
                )
                cost_arcs = self.compute_cost(
                    out_param, tgt_param_arcs, slice(8, 14), device
                )

        cost_bbox[:, tgt_ids == 0] += cost_lines
        cost_bbox[:, tgt_ids == 1] += cost_circles
        cost_bbox[:, tgt_ids == 2] += cost_arcs * 2 / 3 # arcs have 6 params instead of 4

        # Compute the giou cost between boxes
        if self.cost_giou > 0:
            print("computing giou")
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_param[..., 14:18]),
                box_cxcywh_to_xyxy(tgt_param[..., 14:18]),
            )
            C = (
                self.cost_bbox * cost_bbox
                + self.cost_class * cost_class
                + self.cost_giou * cost_giou
            )
        else:
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class

        # Final cost matrix

        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["parameters"]) for v in targets]
        indices = [
            linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
        ]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]


def build_matcher(args):
    assert args.matcher_type in [
        "HungarianMatcher",
    ], "Unknown args.matcher_type: {}".format(args.matcher_type)
    print(f"computing giou in matcher {args.set_cost_giou > 0}")

    return HungarianMatcher(
        cost_class=args.set_cost_class,
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou,
        focal_alpha=args.focal_alpha,
        to_xyxy=args.to_xyxy,
        min_permut_loss=args.min_permut_loss,
    )

# -*- coding: utf-8 -*-
"""
@File    :   visualizer.py
@Time    :   2022/04/05 11:39:33
@Author  :   Shilong Liu 
@Contact :   liusl20@mail.tsinghua.edu.cn; slongliu86@gmail.com
Modified from COCO evaluator
"""

import os
import torch
import numpy as np
import datetime

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Arc
from .box_ops import arc_cxcywh2_to_xy3
from .primitives import get_arc_plot_params


def renorm(
    img: torch.FloatTensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
) -> torch.FloatTensor:
    # img: tensor(3,H,W) or tensor(B,3,H,W)
    # return: same as img
    assert img.dim() == 3 or img.dim() == 4, (
        "img.dim() should be 3 or 4 but %d" % img.dim()
    )
    if img.dim() == 3:
        assert img.size(0) == 3, 'img.size(0) should be 3 but "%d". (%s)' % (
            img.size(0),
            str(img.size()),
        )
        img_perm = img.permute(1, 2, 0)
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        img_res = img_perm * std + mean
        return img_res.permute(2, 0, 1)
    else:  # img.dim() == 4
        assert img.size(1) == 3, 'img.size(1) should be 3 but "%d". (%s)' % (
            img.size(1),
            str(img.size()),
        )
        img_perm = img.permute(0, 2, 3, 1)
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        img_res = img_perm * std + mean
        return img_res.permute(0, 3, 1, 2)


class ColorMap:
    def __init__(self, basergb=[255, 255, 0]):
        self.basergb = np.array(basergb)

    def __call__(self, attnmap):
        # attnmap: h, w. np.uint8.
        # return: h, w, 4. np.uint8.
        assert attnmap.dtype == np.uint8
        h, w = attnmap.shape
        res = self.basergb.copy()
        res = res[None][None].repeat(h, 0).repeat(w, 1)  # h, w, 3
        attn1 = attnmap.copy()[..., None]  # h, w, 1
        res = np.concatenate((res, attn1), axis=-1).astype(np.uint8)
        return res


from PIL import Image


def is_large_arc(rad_angle):
    if rad_angle[0] <= np.pi:
        return not (rad_angle[0] < rad_angle[1] < (np.pi + rad_angle[0]))
    return (rad_angle[0] - np.pi) < rad_angle[1] < rad_angle[0]


class COCOVisualizer:
    def __init__(self) -> None:
        pass

    def visualize(
        self,
        img,
        tgt,
        caption=None,
        dpi=120,
        savedir=None,
        img_name=None,
        show_text=False,
        primitives_to_show=["line", "circle", "arc"],
        show_boxes=True,
        show_image=True,
        fixed_colors=True,
        ax=None,
        linewidth=2,
    ):
        """
        img: tensor(3, H, W)
        tgt: make sure they are all on cpu.
            must have items: 'image_id', 'boxes', 'size'
        """

        if ax is None:
            plt.figure(dpi=dpi)
            plt.rcParams["font.size"] = "5"
            ax = plt.gca()
        plt.axis("off")
        img = renorm(img).permute(1, 2, 0)
        if show_image:
            ax.imshow(img)
        else:
            ax.set_xlim([0, img.shape[1]])
            ax.set_ylim([img.shape[0], 0])
            ax.set_aspect("equal", adjustable="box")

        self.addtgt(
            tgt,
            show_text=show_text,
            primitives_to_show=primitives_to_show,
            show_boxes=show_boxes,
            with_color=show_image,
            ax=ax,
            fixed_colors=fixed_colors,
            linewidth=linewidth,
        )

        if savedir is not None:
            if img_name is None:
                date = str(datetime.datetime.now()).replace(" ", "-")
                img_id = int(tgt["image_id"])
                if caption is None:
                    savename = f"{savedir}/{img_id}-{date}.png"
                else:
                    savename = f"{savedir}/{caption}-{img_id}-{date}.png"
            else:
                savename = f"{savedir}/{img_name}"
            print(f"savename: {savename}")

            os.makedirs(os.path.dirname(savename), exist_ok=True)
            plt.axis("off")
            plt.savefig(savename, bbox_inches="tight", pad_inches=0)
            plt.close()

    def addtgt(
        self,
        tgt,
        show_text=False,
        show_boxes=True,
        primitives_to_show=["line", "circle", "arc"],
        with_color=True,
        ax=None,
        fixed_colors=True,
        linewidth=2,
    ):
        """
        - tgt: dict. args:
            - boxes: num_boxes, 4. xywh, [0,1].
            - box_label: num_boxes.
        """
        #
        H, W = tgt["size"].tolist()
        line_k, circle_k, arc_k = 0, 0, 0
        if "colors" in tgt.keys():
            colors = tgt["colors"]
            fixed_colors = False
        else:
            colors = None

        for k, param in enumerate(tgt["parameters"].cpu()):
            _string = str(tgt["labels"][k])
            box = param[14:]
            unnormbbox = box * torch.Tensor([W, H, W, H])
            unnormbbox[:2] -= unnormbbox[2:] / 2
            [bbox_x, bbox_y, bbox_w, bbox_h] = unnormbbox.tolist()
            poly = [
                [bbox_x, bbox_y],
                [bbox_x, bbox_y + bbox_h],
                [bbox_x + bbox_w, bbox_y + bbox_h],
                [bbox_x + bbox_w, bbox_y],
            ]
            if colors is None:
                c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
            else:
                c = colors[k]

            if ("line" in _string) and ("line" in primitives_to_show):
                if fixed_colors:
                    c = "green"
                line = param[:4]  # cxcywh
                unnormline = line * torch.Tensor([W, H, W, H])
                unnormline[:2] -= unnormline[2:] / 2  # xywh
                [bbox_x, bbox_y, bbox_w, bbox_h] = unnormline.tolist()
                ax.plot(
                    [bbox_x, bbox_x + bbox_w],
                    [bbox_y, bbox_y + bbox_h],
                    c=c,
                    linewidth=linewidth,
                )
                bbox_y = bbox_y + bbox_h

            elif ("circle" in _string) and ("circle" in primitives_to_show):
                if fixed_colors:
                    c = "royalblue"

                circle = param[4:8] # center, radius, center norm, radius norm
                unnormcircle = circle * torch.Tensor([W, H, W, H])
                unnormcircle[:2] -= unnormcircle[2:] / 2
                [bbox_x, bbox_y, bbox_w, bbox_h] = unnormcircle.tolist()
                center = (bbox_x + bbox_w / 2, bbox_y + bbox_h / 2)
                radius = bbox_w / 2
                circle = Circle(center, radius, fill=None, color=c, linewidth=linewidth)
                ax.add_patch(circle)

            elif ("arc" in _string) and ("arc" in primitives_to_show):
                if fixed_colors:
                    c = "firebrick"
                arc = param[8:14] # start point, end point, mid-point
                unnorm_arc = arc * torch.Tensor([W, H, W, H, W, H])
                arc = arc_cxcywh2_to_xy3(unnorm_arc) # start point, end point, mid-point
                arc = np.array(arc.tolist())
                theta1, theta_mid, theta2, c_xy, diameter = get_arc_plot_params(arc)
                ax.scatter(arc[0], arc[1], s=7 * linewidth**2, c=c, marker="^")
                ax.scatter(arc[2], arc[3], s=7 * linewidth**2, c=c, marker="v")
                ax.scatter(arc[4], arc[5], s=7 * linewidth**2, c=c, marker="o")

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
                        linewidth=linewidth,
                    )
                ax.add_patch(arc_patch_2)

            if show_boxes and _string in primitives_to_show:
                np_poly = np.array(poly).reshape((4, 2))
                ax.add_patch(Polygon(np_poly, fill=None, edgecolor=c, linewidth=2))

            if show_text and _string in primitives_to_show:
                if "scores" in tgt:
                    label_score = f"{_string} {tgt['scores'][k]:.2f} "
                else:
                    label_score = _string
                ax.text(
                    bbox_x,
                    bbox_y,
                    label_score,
                    color="black",
                    bbox={"facecolor": c, "alpha": 0.6, "pad": 1},
                    fontsize=6 * linewidth,
                )

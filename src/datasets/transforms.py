"""
Transforms and data augmentation for both image + primitives.

modified based on https://github.com/mlpc-ucsd/LETR/blob/master/src/datasets/transforms.py which is
modified based on https://github.com/facebookresearch/detr/blob/master/datasets/transforms.py
"""

import random
import matplotlib.pyplot as plt
import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numbers
import warnings
from typing import Tuple, List, Optional
from PIL import Image
from torch import Tensor
import math
from util.box_ops import (
    arc_xy3_to_cxcywh2,
    arc_cxcywh2_to_xy3,
    box_xyxy_to_cxcywh,
    box_xyxy_to_cxcywh_abs,
    get_box_from_arcs,
)
import numpy as np

PRIMITIVES = ["lines", "circles", "arcs"]


def get_default_bbox(circles):
    if circles.shape[0] == 0:
        return circles
    condition1 = circles[:, 0] > circles[:, 2]
    if condition1.any():
        indices1 = torch.where(condition1)
        circles[indices1] = circles[indices1][:, [2, 1, 0, 3]]

    condition2 = circles[:, 1] > circles[:, 3]
    if condition2.any():
        indices2 = torch.where(condition2)

        circles[indices2] = circles[indices2][:, [0, 3, 2, 1]]

    return circles


def get_default_arc(arcs):  # TODO: reverse these
    if arcs.shape[0] == 0:
        return arcs
    condition1 = arcs[:, 0] > arcs[:, 2]
    if condition1.any():
        indices1 = torch.where(condition1)
        arcs[indices1] = arcs[indices1][:, [2, 3, 0, 1, 4, 5]]

    condition2 = (arcs[:, 1] > arcs[:, 3]) & (arcs[:, 0] == arcs[:, 2])
    if condition2.any():
        indices2 = torch.where(condition2)

        arcs[indices2] = arcs[indices2][:, [2, 3, 0, 1, 4, 5]]

    return arcs


def get_out_of_bounds_mask(cropped_primitive, h, w):
    remove_x_min = torch.logical_and(
        cropped_primitive[:, 0] < 0, cropped_primitive[:, 2] < 0
    )
    remove_x_max = torch.logical_and(
        cropped_primitive[:, 0] > w, cropped_primitive[:, 2] > w
    )
    remove_x = torch.logical_or(remove_x_min, remove_x_max)
    keep_x = ~remove_x

    remove_y_min = torch.logical_and(
        cropped_primitive[:, 1] < 0, cropped_primitive[:, 3] < 0
    )
    remove_y_max = torch.logical_and(
        cropped_primitive[:, 1] > h, cropped_primitive[:, 3] > h
    )
    remove_y = torch.logical_or(remove_y_min, remove_y_max)
    keep_y = ~remove_y

    keep = torch.logical_and(keep_x, keep_y)
    return keep


def instance_aware_crop(image, target):
    h, w = target["size"]

    if "circles" in target:
        x_borders_circles, y_borders_circles = (
            target["circles"][:, [0, 2]],
            target["circles"][:, [1, 3]],
        )
        min_circle_x, max_circle_x = x_borders_circles.min(), x_borders_circles.max()
        min_circle_y, max_circle_y = y_borders_circles.min(), y_borders_circles.max()
    else:
        min_circle_x, min_circle_y = w, h
        max_circle_x, max_circle_y = torch.tensor(0), torch.tensor(0)

    if "lines" in target:
        x_borders_lines, y_borders_lines = (
            target["lines"][:, [0, 2]],
            target["lines"][:, [1, 3]],
        )
        min_line_x, max_line_x = x_borders_lines.min(), x_borders_lines.max()
        min_line_y, max_line_y = y_borders_lines.min(), y_borders_lines.max()
    else:
        min_line_x, min_line_y = w, h
        max_line_x, max_line_y = torch.tensor(0), torch.tensor(0)

    if ("arcs" in target) and (target["arcs"].shape[0] > 0):
        borders_x = target["arcs"][:, [0, 2, 4]]
        borders_y = target["arcs"][:, [1, 3, 5]]
        min_arc_x, min_arc_y = borders_x.min(), borders_y.min()
        max_arc_x, max_arc_y = borders_x.max(), borders_y.max()
    else:
        max_arc_x, max_arc_y = torch.tensor(0), torch.tensor(0)
        min_arc_x, min_arc_y = w, h

    min_border_x = torch.min(
        torch.stack([min_circle_x, min_line_x, min_arc_x]),
        dim=0,
    ).values
    min_border_y = torch.min(
        torch.stack([min_circle_y, min_line_y, min_arc_y]),
        dim=0,
    ).values
    max_border_x = torch.max(
        torch.stack([max_circle_x, max_line_x, max_arc_x]),
        dim=0,
    ).values
    max_border_y = torch.max(
        torch.stack([max_circle_y, max_line_y, max_arc_y]),
        dim=0,
    ).values

    random_center_offset = random.randint(0, 30)

    i, j = int(max(min_border_y.item() - random_center_offset, 0)), int(
        max(min_border_x.item() - random_center_offset, 0)
    )

    h = int(min(max_border_y.item() - i + random.randint(0, 30), h - i))
    w = int(min(max_border_x.item() - j + random.randint(0, 30), w - j))
    region = (i, j, h, w)

    cropped_image = F.crop(image, *region)
    target = target.copy()
    target["size"] = torch.tensor([h, w])
    target["circles"] = target["circles"] - torch.as_tensor(
        [j, i, j, i], dtype=torch.float32
    )
    target["lines"] = target["lines"] - torch.as_tensor(
        [j, i, j, i], dtype=torch.float32
    )
    target["arcs"] = target["arcs"] - torch.as_tensor(
        [j, i, j, i, j, i], dtype=torch.float32
    )

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()

    if "lines" in target:
        lines = target["lines"]
        lines = lines[:, [2, 3, 0, 1]] * torch.as_tensor(
            [-1, 1, -1, 1]
        ) + torch.as_tensor([w, 0, w, 0])
        target["lines"] = lines
    if "circles" in target:
        circles = target["circles"]
        circles = circles * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor(
            [w, 0, w, 0]
        )

        target["circles"] = get_default_bbox(circles)
    if "arcs" in target:
        arcs = target["arcs"]
        arcs = arcs * torch.as_tensor([-1, 1, -1, 1, -1, 1]) + torch.as_tensor(
            [w, 0, w, 0, w, 0]
        )
        target["arcs"] = get_default_arc(arcs)

    return flipped_image, target


def vflip(image, target):
    flipped_image = F.vflip(image)

    w, h = image.size

    target = target.copy()

    if "lines" in target:
        lines = target["lines"]
        lines = lines * torch.as_tensor([1, -1, 1, -1]) + torch.as_tensor([0, h, 0, h])
        vertical_line_idx = lines[:, 0] == lines[:, 2]
        lines[vertical_line_idx] = torch.index_select(
            lines[vertical_line_idx], 1, torch.tensor([2, 3, 0, 1])
        )
        target["lines"] = lines
    if "circles" in target:
        circles = target["circles"]
        circles = circles * torch.as_tensor([1, -1, 1, -1]) + torch.as_tensor(
            [0, h, 0, h]
        )

        target["circles"] = get_default_bbox(circles)
    if "arcs" in target:
        arcs = target["arcs"]
        arcs = arcs * torch.as_tensor([1, -1, 1, -1, 1, -1]) + torch.as_tensor(
            [0, h, 0, h, 0, h]
        )
        target["arcs"] = get_default_arc(arcs)

    return flipped_image, target


def ccw_rotation(image, target):
    rotateded_image = F.rotate(image, 90, expand=True)
    w, h = rotateded_image.size
    # print("inside ccw rotation")
    target = target.copy()

    target["size"] = torch.tensor([h, w])

    if "lines" in target:
        lines = target["lines"]
        lines = lines[:, [1, 0, 3, 2]] * torch.as_tensor(
            [1, -1, 1, -1]
        ) + torch.as_tensor([0, h, 0, h])

        x_switch_idx = lines[:, 0] > lines[:, 2]
        lines[x_switch_idx] = torch.index_select(
            lines[x_switch_idx], 1, torch.tensor([2, 3, 0, 1])
        )

        y_switch_idx = torch.logical_and(
            lines[:, 0] == lines[:, 2], lines[:, 1] > lines[:, 3]
        )
        lines[y_switch_idx] = torch.index_select(
            lines[y_switch_idx], 1, torch.tensor([2, 3, 0, 1])
        )

        target["lines"] = lines
    if "circles" in target:
        circles = target["circles"]
        circles = circles[:, [1, 0, 3, 2]] * torch.as_tensor(
            [1, -1, 1, -1]
        ) + torch.as_tensor([0, h, 0, h])

        x_switch_idx = circles[:, 0] > circles[:, 2]
        circles[x_switch_idx] = torch.index_select(
            circles[x_switch_idx], 1, torch.tensor([2, 3, 0, 1])
        )

        y_switch_idx = torch.logical_and(
            circles[:, 0] == circles[:, 2], circles[:, 1] > circles[:, 3]
        )
        circles[y_switch_idx] = torch.index_select(
            circles[y_switch_idx], 1, torch.tensor([2, 3, 0, 1])
        )

        target["circles"] = get_default_bbox(circles)

    if "arcs" in target:
        arcs = target["arcs"]
        arcs = arcs[:, [1, 0, 3, 2, 5, 4]] * torch.as_tensor(
            [1, -1, 1, -1, 1, -1]
        ) + torch.as_tensor([0, h, 0, h, 0, h])
        target["arcs"] = get_default_arc(arcs)

    # target["boxes6d"] = torch.cat([target[primitive] for primitive in PRIMITIVES])
    return rotateded_image, target


def cw_rotation(image, target):
    rotateded_image = F.rotate(image, -90, expand=True)
    w, h = rotateded_image.size
    target = target.copy()
    target["size"] = torch.tensor([h, w])

    if "lines" in target:
        lines = target["lines"]
        lines = lines[:, [1, 0, 3, 2]] * torch.as_tensor(
            [-1, 1, -1, 1]
        ) + torch.as_tensor([w, 0, w, 0])

        # in dataset, we assume the first point is the left point
        x_switch_idx = lines[:, 0] > lines[:, 2]
        lines[x_switch_idx] = torch.index_select(
            lines[x_switch_idx], 1, torch.tensor([2, 3, 0, 1])
        )

        # in dataset, if two points have same x coord, we assume the first point is the upper point
        y_switch_idx = torch.logical_and(
            lines[:, 0] == lines[:, 2], lines[:, 1] > lines[:, 3]
        )
        lines[y_switch_idx] = torch.index_select(
            lines[y_switch_idx], 1, torch.tensor([2, 3, 0, 1])
        )

        target["lines"] = lines
    if "circles" in target:
        circles = target["circles"]
        circles = circles[:, [1, 0, 3, 2]] * torch.as_tensor(
            [-1, 1, -1, 1]
        ) + torch.as_tensor([w, 0, w, 0])

        # in dataset, we assume the first point is the left point
        x_switch_idx = circles[:, 0] > circles[:, 2]
        circles[x_switch_idx] = torch.index_select(
            circles[x_switch_idx], 1, torch.tensor([2, 3, 0, 1])
        )

        # in dataset, if two points have same x coord, we assume the first point is the upper point
        y_switch_idx = torch.logical_and(
            circles[:, 0] == circles[:, 2], circles[:, 1] > circles[:, 3]
        )
        circles[y_switch_idx] = torch.index_select(
            circles[y_switch_idx], 1, torch.tensor([2, 3, 0, 1])
        )

        target["circles"] = get_default_bbox(circles)
    if "arcs" in target:
        arcs = target["arcs"]
        arcs = arcs[:, [1, 0, 3, 2, 5, 4]] * torch.as_tensor(
            [-1, 1, -1, 1, -1, 1]
        ) + torch.as_tensor([w, 0, w, 0, w, 0])
        target["arcs"] = get_default_arc(arcs)

    # target["boxes6d"] = torch.cat([target[primitive] for primitive in PRIMITIVES])

    return rotateded_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(
        float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size)
    )
    ratio_width, ratio_height = ratios

    target = target.copy()

    for primitive in ["lines", "circles"]:
        if primitive in target:
            primitive_annotation = target[primitive]
            scaled_primitive_anno = primitive_annotation * torch.as_tensor(
                [ratio_width, ratio_height, ratio_width, ratio_height]
            )
            target[primitive] = scaled_primitive_anno

    if "arcs" in target:
        arc_annotation = target["arcs"]
        scaled_arc_anno = arc_annotation * torch.as_tensor(
            [
                ratio_width,
                ratio_height,
                ratio_width,
                ratio_height,
                ratio_width,
                ratio_height,
            ]
        )
        target["arcs"] = scaled_arc_anno

    h, w = size
    target["size"] = torch.tensor([h, w])
    # target["boxes6d"] = torch.cat([target[primitive] for primitive in PRIMITIVES])
    return rescaled_image, target


class InstanceAwareCrop(object):
    def __init__(self):
        pass

    def __call__(self, img, target):
        return instance_aware_crop(img, target)


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int, plot_example=False):
        self.min_size = min_size
        self.max_size = max_size
        self.plot_example = plot_example

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region, self.plot_example)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.0))
        crop_left = int(round((image_width - crop_width) / 2.0))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return vflip(img, target)
        return img, target


class RandomCounterClockwiseRotation(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return ccw_rotation(img, target)
        return img, target


class RandomClockwiseRotation(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return cw_rotation(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomErasing(object):
    def __init__(
        self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, fill=False
    ):
        if not isinstance(value, (numbers.Number, str, tuple, list)):
            raise TypeError(
                "Argument value should be either a number or str or a sequence"
            )
        if isinstance(value, str) and value != "random":
            raise ValueError("If value is str, it should be 'random'")
        if not isinstance(scale, (tuple, list)):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, (tuple, list)):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("Scale should be between 0 and 1")
        if p < 0 or p > 1:
            raise ValueError("Random erasing probability should be between 0 and 1")

        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    @staticmethod
    def get_params(
        img: Tensor,
        scale: Tuple[float, float],
        ratio: Tuple[float, float],
        value: Optional[List[float]] = None,
    ) -> Tuple[int, int, int, int, Tensor]:
        if isinstance(img, Tensor):
            img_c, img_h, img_w = img.shape[-3], img.shape[-2], img.shape[-1]
            area = img_h * img_w
        elif isinstance(img, Image.Image):
            img_c = 3
            img_w, img_h = img.size
            area = img_h * img_w
        else:
            raise TypeError("img is not type Tensor or Image")

        for _ in range(10):
            erase_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.empty(1).uniform_(ratio[0], ratio[1]).item()

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))
            if not (h < img_h and w < img_w):
                continue

            if value is None:
                v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
            else:
                v = torch.tensor(value)[:, None, None]

            i = torch.randint(0, img_h - h + 1, size=(1,)).item()
            j = torch.randint(0, img_w - w + 1, size=(1,)).item()
            return i, j, h, w, v

        # Return original image
        return 0, 0, img_h, img_w, img

    def __call__(self, img, target):
        i, j, h, w, v = RandomErasing.get_params(img, self.scale, self.ratio)
        img_tensor = torch.tensor(np.transpose(np.asarray(img), (2, 0, 1)))
        new_img = F.erase(img_tensor, i, j, h, w, v)
        new_img = new_img.numpy()
        new_img = Image.fromarray(new_img)
        return new_img, target


class ColorJitter(object):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4):
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(
            hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False
        )

    def _check_input(
        self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True
    ):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    "If {} is a single number, it must be non negative.".format(name)
                )
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError(
                "{} should be a single number or a list/tuple with lenght 2.".format(
                    name
                )
            )

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def __call__(self, img, target):
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = (
                    torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                )
                img = F.adjust_brightness(img, brightness_factor)

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = (
                    torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                )
                img = F.adjust_contrast(img, contrast_factor)

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = (
                    torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                )
                img = F.adjust_saturation(img, saturation_factor)

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = F.adjust_hue(img, hue_factor)

        return img, target


class RandomSelect(object):
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class Normalize(object):
    def __init__(self, mean, std, boxes_only=False):
        self.mean = mean
        self.std = std
        self.boxes_only = boxes_only

    def __call__(self, image, target=None):
        # boxes_only = self.boxes_only
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        all_params = []
        if "lines" in target:
            primitive_annotation = target["lines"]
            primitive_annotation = primitive_annotation / torch.tensor(
                [w, h, w, h], dtype=torch.float32
            )
            target["lines_boxes"] = box_xyxy_to_cxcywh_abs(primitive_annotation)
            target["lines"] = box_xyxy_to_cxcywh(primitive_annotation)
            zeros_tensor = torch.zeros(
                (target["lines"].shape[0], 10), dtype=torch.float32
            )
            full_param = torch.hstack(
                (target["lines"], zeros_tensor, target["lines_boxes"])
            )
            # if boxes_only:
            #     full_param = torch.hstack(
            #         (
            #             torch.zeros(
            #                 (target["lines"].shape[0], 14), dtype=torch.float32
            #             ),
            #             target["lines_boxes"],
            #         )
            #     )
            all_params.append(full_param)

        if "circles" in target:
            primitive_annotation = target["circles"]
            primitive_annotation = primitive_annotation / torch.tensor(
                [w, h, w, h], dtype=torch.float32
            )
            primitive_boxes = get_default_bbox(primitive_annotation)
            target["circles_boxes"] = box_xyxy_to_cxcywh_abs(primitive_boxes)
            target["circles"] = box_xyxy_to_cxcywh(primitive_annotation)
            zeros_tensor_lines = torch.zeros(
                (target["circles"].shape[0], 4), dtype=torch.float32
            )
            zeros_tensor_arcs = torch.zeros(
                (target["circles"].shape[0], 6), dtype=torch.float32
            )
            full_param = torch.hstack(
                (
                    zeros_tensor_lines,
                    target["circles"],
                    zeros_tensor_arcs,
                    target["circles_boxes"],
                )
            )
            # if boxes_only:
            #     full_param = torch.hstack(
            #         (
            #             torch.zeros(
            #                 (target["circles"].shape[0], 14), dtype=torch.float32
            #             ),
            #             target["circles_boxes"],
            #         )
            #     )
            all_params.append(full_param)

        if "arcs" in target:
            arc_annotation = target["arcs"]
            arc_boxes = get_box_from_arcs(arc_annotation)
            arc_boxes = arc_boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["arcs_boxes"] = box_xyxy_to_cxcywh_abs(arc_boxes)
            arc_annotation = arc_annotation / torch.tensor(
                [w, h, w, h, w, h], dtype=torch.float32
            )
            target["arcs"] = arc_xy3_to_cxcywh2(arc_annotation)
            # zeros for lines and circles
            zeros_tensor = torch.zeros(
                (target["arcs"].shape[0], 8), dtype=torch.float32
            )
            full_param = torch.hstack(
                (zeros_tensor, target["arcs"], target["arcs_boxes"])
            )
            # if boxes_only:
            #     full_param = torch.hstack(
            #         (
            #             torch.zeros((target["arcs"].shape[0], 14), dtype=torch.float32),
            #             target["arcs_boxes"],
            #         )
            #     )
            all_params.append(full_param)

        target["parameters"] = torch.cat(all_params)
        target["boxes"] = torch.cat(
            [target[f"{primitive}_boxes"] for primitive in PRIMITIVES]
        )

        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

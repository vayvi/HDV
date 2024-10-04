"""
Modified based on Detr: https://github.com/facebookresearch/detr/blob/master/datasets/coco.py
"""

from pathlib import Path

import torch
import torch.utils.data
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets import CocoDetection as Coco
import numpy as np

from synthetic_module.synthetic import SyntheticDiagram
from synthetic_module import DEFAULT_WIDTH, DEFAULT_HEIGHT, SYNTHETIC_RESRC_PATH

from . import transforms as T
from ..util.box_ops import box_xyxy_to_cxcywh_abs, get_box_from_arcs


class CocoDetectionOnTheFly(VisionDataset):
    """COCO format dataset with support for on the fly generation of synthetic data"""

    def __init__(
        self,
        background_root=SYNTHETIC_RESRC_PATH,
        transform=None,
        target_transform=None,
        transforms=None,
        args=None,
        num_samples=None,
    ) -> None:
        super().__init__(background_root, transforms, transform, target_transform)
        self.num_samples = num_samples
        self.ids = list(range(self.num_samples))

        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask()
        self.args = args

    def _load_image(self, id: int, diagram):
        return diagram.to_image()

    def _load_target(self, id: int, diagram):
        return diagram.get_annotation_on_the_fly()

    def __getitem__(self, index: int):
        id = self.ids[index]
        width, height = DEFAULT_WIDTH, DEFAULT_HEIGHT
        diagram = SyntheticDiagram(width, height)
        image = self._load_image(id, diagram)
        target = self._load_target(id, diagram)

        image_id = self.ids[id]
        target = {"image_id": image_id, "annotations": target}
        image, target = self.prepare(image, target, self.args)

        if self._transforms is not None:
            image, target = self._transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return self.num_samples


class CocoDetection(Coco):
    def __init__(self, img_folder, ann_file, transforms, args):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask()
        self.args = args

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target, self.args)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def transform_circle(circle, args):
    x, y, rx, ry = circle
    return [x - rx, y - ry, x + rx, y + ry]


def get_line_targets(annotation, args):
    lines = np.array([obj["line"] for obj in annotation if "line" in obj])
    lines = torch.as_tensor(lines, dtype=torch.float32).reshape(-1, 4)
    lines_area = torch.abs(
        lines[:, 2] * lines[:, 3]
    )  # this is the bbox area (also the case for circles)
    lines[:, 2:] += lines[:, :2]  # xyxy
    lines_boxes = box_xyxy_to_cxcywh_abs(lines.clone())
    lines_labels = torch.zeros(len(lines), dtype=torch.int64)
    return lines, lines_labels, lines_area, lines_boxes


def get_circle_targets(annotation, args):
    circles = [
        transform_circle(obj["circle"], args) for obj in annotation if "circle" in obj
    ]
    circles = torch.as_tensor(circles, dtype=torch.float32).reshape(-1, 4)
    circles_area = torch.abs(
        (circles[:, 2] - circles[:, 0]) * (circles[:, 3] - circles[:, 1])
    )
    circles = T.get_default_bbox(circles)
    circles_boxes = box_xyxy_to_cxcywh_abs(circles.clone())
    circles_labels = torch.ones(len(circles), dtype=torch.int64)
    return circles, circles_labels, circles_area, circles_boxes


def get_arc_targets(annotation, args):
    arcs = np.array([obj["arc"] for obj in annotation if "arc" in obj]).reshape(-1, 6)

    arcs = torch.as_tensor(arcs, dtype=torch.float32).reshape(-1, 6)
    arcs = T.get_default_arc(arcs)
    arc_labels = torch.full((len(arcs),), 2, dtype=torch.int64)
    arcs_boxes = get_box_from_arcs(arcs)
    arcs_boxes = box_xyxy_to_cxcywh_abs(arcs_boxes)
    arcs_area = arcs_boxes[:, 2] * arcs_boxes[:, 3]

    return arcs, arc_labels, arcs_area, arcs_boxes


class ConvertCocoPolysToMask(object):
    def __call__(self, image, tgt, args):
        w, h = image.size
        image_id = tgt["image_id"]
        image_id = torch.tensor([image_id])
        anno = tgt["annotations"]
        anno = [obj for obj in anno]
        target = {}
        primitives = ["lines", "circles", "arcs"]
        get_targets_functions = [get_line_targets, get_circle_targets, get_arc_targets]
        for primitive, get_targets in zip(primitives, get_targets_functions):
            (
                target[primitive],
                target[f"{primitive}_labels"],
                target[f"{primitive}_area"],
                target[f"{primitive}_boxes"],
            ) = get_targets(anno, args)
        target["labels"] = torch.cat(
            [target[f"{primitive}_labels"] for primitive in primitives]
        )
        target["boxes"] = torch.cat(
            [target[f"{primitive}_boxes"] for primitive in primitives]
        )
        target["image_id"] = image_id
        target["iscrowd"] = torch.zeros(len(anno))
        target["area"] = torch.cat(
            [target[f"{primitive}_area"] for primitive in primitives]
        )
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, args):
    normalize = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                [0.538, 0.494, 0.453],
                [0.257, 0.263, 0.273],
                args.boxes_only,
            ),
        ]
    )

    scales = args.data_aug_scales or [512, 544, 576, 608, 640, 672, 680, 690, 704, 736, 768, 788, 800]
    test_size = 1100
    # maximal size of the longer image side (reduce to prevent CUDA out of memory)
    max_size = args.data_aug_max_size or 1333

    if args.eval:
        return T.Compose(
            [
                T.InstanceAwareCrop(),
                T.RandomResize([test_size], max_size=max_size),
                normalize,
            ]
        )
    else:
        if image_set == "train":
            return T.Compose(
                [
                    T.RandomSelect(
                        T.RandomHorizontalFlip(),
                        T.RandomVerticalFlip(),
                    ),
                    T.RandomSelect(
                        T.RandomClockwiseRotation(),
                        T.RandomCounterClockwiseRotation(),
                    ),
                    T.RandomResize([500, 600]),
                    T.InstanceAwareCrop(),
                    T.RandomResize(scales, max_size=max_size),
                    normalize,
                ]
            )
        if image_set == "val":
            return T.Compose(
                [
                    T.RandomResize([test_size], max_size=max_size),
                    normalize,
                ]
            )

        raise ValueError(f"unknown {image_set}")


def build(image_set, args, mode="primitives"):
    root = Path(args.coco_path)
    if not args.on_the_fly:
        assert root.exists(), f"provided COCO path {root} does not exist"

    if args.on_the_fly and ("train" in image_set):
        dataset = CocoDetectionOnTheFly(
            transforms=make_coco_transforms(image_set, args),
            args=args,
            num_samples=args.num_samples,
        )
        return dataset
    elif hasattr(args, 'on_the_fly_val') and args.on_the_fly_val and ("val" in image_set):
        dataset = CocoDetectionOnTheFly(
            transforms=make_coco_transforms(image_set, args),
            args=args,
            num_samples=args.num_samples // 5,
        )
        return dataset

    # PATHS = {
    #     "train": (root / "train", root / "annotations" / f"{mode}_train.json"),
    #     "val": (root / "val", root / "annotations" / f"{mode}_val.json"),
    # }
    #
    # img_folder, ann_file = PATHS["train" if "train" in image_set else "val"]
    #
    # print("#######", img_folder, "~####################")
    # print("#######", ann_file, "~###########")
    #
    # dataset = CocoDetection(
    #     img_folder,
    #     ann_file,
    #     transforms=make_coco_transforms(image_set, args),
    #     args=args,
    # )

    dataset = CocoDetection(
        root / image_set,
        root / "annotations" / f"{mode}_{image_set}.json",
        transforms=make_coco_transforms(image_set, args),
        args=args,
    )
    return dataset

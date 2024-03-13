from .helper.image import paste_with_blured_borders, resize
from .helper.path import coerce_to_path_and_check_exist
import numpy as np
from PIL import Image, ImageFilter
import cv2
import os
from numpy.random import uniform, choice
from random import randint, shuffle
from .background import BackgroundElement
from .diagram import DiagramElement
from .element import GlyphElement, GLYPH_COLORED_FREQ
from .helper.noise import get_random_noise_pattern
from .helper.seed import use_seed
import json
from tqdm.notebook import tqdm_notebook
import PIL
from .helper.color_utils import rgb_to_gray, img_8bit_to_float, gray_float_to_8bit
import imageio
import matplotlib.pyplot as plt
from synthetic_module import DEFAULT_HEIGHT, DEFAULT_WIDTH
import random
from synthetic_module import background

MAX_WIDTH, MAX_HEIGHT = 3 * DEFAULT_WIDTH, 3 * DEFAULT_HEIGHT
MIN_WIDTH, MIN_HEIGHT = 0.5 * DEFAULT_WIDTH, 0.5 * DEFAULT_HEIGHT
BLACK_AND_WHITE_FREQ = 0.1
BLUR_RADIUS_RANGE = (0.1, 1.5)
BACKGROUND_BLURED_BORDER_WIDTH_RANGE = (1, 10)
LAYOUT_RANGE = {
    "nb_noise_patterns": (0, 5),
    "nb_words": (0, 10),
    "margin_h": (20, 60),
    "margin_v": (20, 60),
    "padding_h": (5, 80),
    "padding_v": (5, 80),
    "caption_padding_v": (0, 20),
    "context_margin_h": (0, 300),
    "context_margin_v": (0, 200),
}
NOISE_STD = 10
NOISE_PATTERN_RANGE = (0, 6)
SCALE_METHOD = [
    PIL.Image.NEAREST,
    PIL.Image.BOX,
    PIL.Image.BILINEAR,
    PIL.Image.HAMMING,
    PIL.Image.BICUBIC,
    PIL.Image.LANCZOS,
]


def lines_xyxy_to_xywh(lines):
    # from x,y,x,y to x,y,dx, dy. order: top point > bottom point, if same y coordinate, right point > left point
    lines = np.array(lines)
    mask = lines[:, 0] > lines[:, 2]

    lines[mask, 0], lines[mask, 2] = lines[mask, 2], lines[mask, 0]
    lines[mask] = lines[mask][:, [2, 3, 0, 1]]
    mask2 = (lines[:, 0] == lines[:, 2]) & (lines[:, 1] > lines[:, 3])
    lines[mask2] = lines[mask2][:, [2, 3, 0, 1]]
    new_lines_pairs = np.column_stack(
        (
            lines[:, 0],
            lines[:, 1],
            lines[:, 2] - lines[:, 0],
            lines[:, 3] - lines[:, 1],
        )
    )  # x,y,dx,dy
    new_lines_pairs = new_lines_pairs.reshape(-1, 2, 2)
    return new_lines_pairs


def circles_to_xywh(circles=None):
    centers, radii = circles
    centers = np.array(centers)
    radii = np.array(radii)
    new_circles = np.column_stack((centers[:, 0], centers[:, 1], radii, radii))
    return new_circles


def xyxy_to_xywh(lines=None, circles=None):
    # changes the format from x,y,x,y to x,y,dx, dy
    # order: top point > bottom point
    if lines is not None:
        lines = np.array(lines)
        lines = lines.reshape(-1, 4)  # xyxy
        mask = lines[:, 0] > lines[:, 2]

        lines[mask, 0], lines[mask, 2] = lines[mask, 2], lines[mask, 0]
        lines[mask] = lines[mask][:, [2, 3, 0, 1]]
        mask2 = (lines[:, 0] == lines[:, 2]) & (lines[:, 1] > lines[:, 3])
        lines[mask2] = lines[mask2][:, [2, 3, 0, 1]]
        new_lines_pairs = np.column_stack(
            (
                lines[:, 0],
                lines[:, 1],
                lines[:, 2] - lines[:, 0],
                lines[:, 3] - lines[:, 1],
            )
        )  # x,y,dx,dy

        new_lines_pairs = new_lines_pairs.reshape(-1, 2, 2)
    else:
        new_lines_pairs = []
    if circles is not None:
        centers, radii = circles
        centers = np.array(centers)
        radii = np.array(radii)
        new_circles = np.column_stack((centers[:, 0], centers[:, 1], radii, radii))
    else:
        new_circles = []
    return new_lines_pairs, new_circles


GLYPH_FREQ = 0.1


class SyntheticDiagram:
    @use_seed()
    def __init__(
        self,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        img_size=None,
    ) -> None:

        # self.median_blur_kernel = choice([0, 3, 5]) # this makes some circles almost disappear or partially disappear
        self.blur_radius = np.random.uniform(*BLUR_RADIUS_RANGE)
        # self.add_noise = choice([True, False], p=[0.8, 0.2])
        self.add_gaussian_noise = choice([True, False], p=[0.8, 0.2])
        self.add_random_erasing = choice([True, False], p=[0.8, 0.2])

        # self.add_resize_noise = choice([True, False], p=[0.8, 0.2])

        # self.smooth = choice([True, False], p=[0.8, 0.2])
        self.add_resize_noise = True
        self.smooth = True
        self.add_gaussian_noise = True
        self.black_and_white = choice(
            [True, False], p=[BLACK_AND_WHITE_FREQ, 1 - BLACK_AND_WHITE_FREQ]
        )
        self.background = BackgroundElement(width, height)
        self.noise_patterns = self._generate_random_noise_patterns()
        if img_size is not None:
            width, height = resize(
                Image.new("L", (width, height)), img_size, keep_aspect_ratio=True
            ).size  # TODO: understand this line
        self.width, self.height = width, height
        margin_h = randint(*LAYOUT_RANGE["margin_h"])
        margin_v = randint(*LAYOUT_RANGE["margin_v"])
        self.diagram_position = (margin_h, margin_v)
        self.diagram = DiagramElement(
            self.width, self.height, diagram_position=self.diagram_position
        )
        self.glyph = None
        if choice([True, False], p=[GLYPH_FREQ, 1 - GLYPH_FREQ]):
            self.glyph = GlyphElement(self.width, self.height)

    @use_seed()
    def _generate_random_noise_patterns(self):
        patterns, positions = [], []
        bg_width, bg_height = self.background.size
        for _ in range(randint(*NOISE_PATTERN_RANGE)):
            pattern, hue_color, value_ratio, position = get_random_noise_pattern(
                bg_width, bg_height
            )
            position = (position[0], position[1])
            patterns.append((pattern, hue_color, value_ratio))
            positions.append(position)
        return patterns, positions

    @property
    def size(self):
        return (self.width, self.height)

    def draw_noise_patterns(self, canvas):
        for (noise, hue_color, value_ratio), pos in zip(*self.noise_patterns):
            x, y = pos
            width, height = noise.size
            patch = np.array(canvas.crop([x, y, x + width, y + height]))
            patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
            patch_hsv[:, :, 0] = hue_color
            patch_hsv[:, :, 2] = patch_hsv[:, :, 2] * value_ratio
            new_patch = Image.fromarray(cv2.cvtColor(patch_hsv, cv2.COLOR_HSV2RGB))
            canvas.paste(new_patch, pos, mask=noise)

    @use_seed()
    def to_image(self, path=None):
        canvas = Image.new(mode="RGB", size=self.size)
        background_img = self.background.to_image()

        paste_with_blured_borders(
            canvas,
            background_img,
            (0, 0),
            randint(*BACKGROUND_BLURED_BORDER_WIDTH_RANGE),
        )
        if self.glyph is not None:
            glyph_img = self.glyph.to_image()
            canvas.paste(glyph_img, (0, 0), mask=glyph_img)
        diagram_img = self.diagram.to_image()
        canvas.paste(diagram_img, (0, 0), mask=diagram_img)
        if self.add_random_erasing and not self.diagram.fill:
            params = {
                "content_width": self.diagram.content_width,
                "content_height": self.diagram.content_height,
                "number_circles": 0,
                "number_words": 0,
                "thickness_range": [1, 2],
            }
            mask_diagram = DiagramElement(
                self.width,
                self.height,
                diagram_position=self.diagram_position,
                **params,
            ).to_image()
            canvas.paste(background_img, (0, 0), mask=mask_diagram)
        if self.add_resize_noise:
            scale_method = SCALE_METHOD[np.random.randint(0, 6)]
            # factor = np.random.uniform(0.6, 1.5)
            factor = np.random.uniform(0.6, 0.9)

            small = canvas.resize(
                (
                    int(factor * self.width),
                    int(factor * self.height),
                ),
                scale_method,
            )
            scale_method = SCALE_METHOD[np.random.randint(0, 6)]
            same = small.resize((self.width, self.height), scale_method)
            repeated_noise = np.repeat(
                np.random.normal(loc=1, scale=0.008, size=(self.height, self.width))[
                    :, :, np.newaxis
                ],
                3,
                axis=2,
            )

            same = same * repeated_noise
            same = np.array(same).astype("uint8")
            canvas = Image.fromarray(same)

        if self.add_gaussian_noise:

            def add_gaussian_noise(img):
                img_arr = np.array(img)
                noisy_img_arr = img_arr + np.random.normal(0, NOISE_STD, img_arr.shape)
                noisy_img_arr = np.clip(noisy_img_arr, 0, 255).astype(np.uint8)
                noisy_img = Image.fromarray(noisy_img_arr)
                return noisy_img

            canvas = add_gaussian_noise(canvas)

        if self.blur_radius > 0:
            canvas = canvas.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))

        if self.smooth:
            canvas = canvas.filter(ImageFilter.SMOOTH)
        if self.black_and_white:
            canvas = canvas.convert("L").convert("RGB")
        if path is not None:
            canvas.save(path)

        return canvas

    def save(self, name, output_dir):
        self._save_to_image(name, output_dir)

    def _save_to_image(self, name, output_dir):
        output_dir = coerce_to_path_and_check_exist(output_dir)
        image_dir = output_dir / "images/"
        os.makedirs(image_dir, exist_ok=True)
        path = image_dir / f"{name}.png"
        img = self.to_image(path)

    def get_annotation(self, name):
        annotation = self.diagram.get_annotation()
        annotation["filename"] = f"{name}.png"
        annotation["height"] = self.height
        annotation["width"] = self.width
        return annotation

    def get_annotation_on_the_fly(self):
        data = self.diagram.get_annotation()
        annotation = []
        if len(data["lines"]) > 0:
            line_set = np.array(data["lines"], dtype=np.float64)
            line_set = lines_xyxy_to_xywh(line_set)
        else:
            line_set = []
        if len(data["circle_centers"]) > 0:
            circle_set = (data["circle_centers"], data["circle_radii"])
            circle_set = circles_to_xywh(circle_set)
        else:
            circle_set = []
        if len(data["arcs"]) > 0:
            arc_set = np.array(data["arcs"], dtype=np.float64)
        else:
            arc_set = np.array([])

        anno_id = 0
        for line in line_set:
            info = {}
            info["id"] = anno_id
            anno_id += 1
            info["category_id"] = 0
            info["line"] = line
            annotation.append(info)

        for circle in circle_set:
            info = {}
            info["id"] = anno_id
            anno_id += 1
            info["category_id"] = 1
            info["circle"] = circle
            annotation.append(info)

        for arc in arc_set:
            info = {}
            info["id"] = anno_id
            anno_id += 1
            info["category_id"] = 2
            info["arc"] = arc  # x0,y0,dx1,dy1, dx_mid, dy_mid
            annotation.append(info)

        # annotation["image_id"] = image_id

        return annotation

    def get_annotation_processed(self, name, image_id, anno_id):
        data = self.diagram.get_annotation()
        annotation = []
        if len(data["lines"]) > 0:
            line_set = np.array(data["lines"], dtype=np.float64)
            line_set = lines_xyxy_to_xywh(line_set)
        else:
            line_set = []
        if len(data["circle_centers"]) > 0:
            circle_set = (data["circle_centers"], data["circle_radii"])
            circle_set = circles_to_xywh(circle_set)
        else:
            circle_set = []
        if len(data["arcs"]) > 0:
            arc_set = np.array(data["arcs"], dtype=np.float64)
        else:
            arc_set = np.array([])
        line_set = line_set.tolist()
        circle_set = circle_set.tolist()
        arc_set = arc_set.tolist()

        for line in line_set:
            info = {}
            info["id"] = anno_id
            anno_id += 1
            info["category_id"] = 0
            info["image_id"] = image_id
            info["line"] = line
            annotation.append(info)

        for circle in circle_set:
            info = {}
            info["id"] = anno_id
            anno_id += 1
            info["category_id"] = 1
            info["image_id"] = image_id
            info["circle"] = circle
            annotation.append(info)

        for arc in arc_set:
            info = {}
            info["id"] = anno_id
            anno_id += 1
            info["category_id"] = 2
            info["image_id"] = image_id
            info["arc"] = arc  # x0,y0,dx1,dy1, dx_mid, dy_mid
            annotation.append(info)
        return annotation, anno_id

    def save_as_val(self, name, output_dir):
        output_dir = coerce_to_path_and_check_exist(output_dir)
        image_dir = output_dir / "val/"
        os.makedirs(image_dir, exist_ok=True)
        path = image_dir / f"{name}.png"
        img = self.to_image(path)


class DiagramsDataset:
    def __init__(
        self,
        num_samples=1000,
        seed=None,
        train_val_split=0.2,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        max_size=1200,
        offset=0,
    ) -> None:
        self.num_samples = num_samples
        self.seed = seed
        self.train_val_split = train_val_split
        self.max_size = max_size  # FIXME: incorporate this
        self.width, self.height = width, height
        self.offset = offset

    def generate_dataset(self, img_size=None, output_dir="synthetic_dataset"):
        os.makedirs(output_dir, exist_ok=True)
        # TODO: fix black and white images
        annotations = []
        output_dir = coerce_to_path_and_check_exist(output_dir)
        if self.offset:
            print("appending to existing dataset")
            assert (
                output_dir / "train.json"
            ).exists(), "training annotations not found"
            assert (
                output_dir / "valid.json"
            ).exists(), "validation annotations not found"
        for k in tqdm_notebook(range(self.offset, self.num_samples + self.offset, 1)):
            name = f"synthetic_diagram_{k}"
            synthetic_diagram = SyntheticDiagram(
                img_size=img_size,
                width=self.width,
                height=self.height,
            )
            synthetic_diagram.save(name, output_dir)
            annotations.append(synthetic_diagram.get_annotation(name))

        shuffle(annotations)
        val_size = int(self.train_val_split * (self.num_samples))
        valid_annotations = annotations[:val_size]
        train_annotations = annotations[val_size:]
        with open(output_dir / "train.json", "w") as json_file:
            json.dump(train_annotations, json_file)
        with open(output_dir / "valid.json", "w") as json_file:
            json.dump(valid_annotations, json_file)

    def generate_val_dataset(
        self, num_val_samples=2, img_size=None, output_dir="synthetic_processed"
    ):
        os.makedirs(output_dir, exist_ok=True)
        anno_dict = {"images": [], "annotations": [], "categories": []}
        anno_dict["categories"] = [
            {"supercategory": "line", "id": "0", "name": "line"},
            {"supercategory": "circle", "id": "1", "name": "circle"},
            {"supercategory": "arc", "id": "2", "name": "arc"},
        ]
        output_dir = coerce_to_path_and_check_exist(output_dir)
        os.makedirs(output_dir / "annotations/", exist_ok=True)
        annotation_path = output_dir / "annotations/primitives_val.json"
        offset = 9000
        anno_id = 0
        for image_id in range(offset, offset + num_val_samples):
            name = f"synthetic_diagram_{image_id}"
            width = int(
                max(min(random.gauss(DEFAULT_WIDTH, 100), MAX_WIDTH), MIN_WIDTH)
            )
            height = int(
                max(min(random.gauss(DEFAULT_HEIGHT, 100), MAX_HEIGHT), MIN_HEIGHT)
            )
            synthetic_diagram = SyntheticDiagram(
                img_size=img_size,
                width=width,
                height=height,
            )
            annotation, anno_id = synthetic_diagram.get_annotation_processed(
                name, image_id, anno_id
            )
            image_info = {
                "file_name": f"{name}.png",
                "height": height,
                "width": width,
                "id": image_id,
            }
            for primitive_annotation in annotation:
                anno_dict["annotations"].append(primitive_annotation)
            anno_dict["images"].append(image_info)
            synthetic_diagram.save_as_val(name, output_dir)
        # return annotations
        with open(annotation_path, "w") as json_file:
            json.dump(anno_dict, json_file)

    def generate_train_dataset(
        self, num_val_samples=2, img_size=None, output_dir="synthetic_processed"
    ):
        os.makedirs(output_dir, exist_ok=True)
        anno_dict = {"images": [], "annotations": [], "categories": []}
        anno_dict["categories"] = [
            {"supercategory": "line", "id": "0", "name": "line"},
            {"supercategory": "circle", "id": "1", "name": "circle"},
            {"supercategory": "arc", "id": "2", "name": "arc"},
        ]
        output_dir = coerce_to_path_and_check_exist(output_dir)
        os.makedirs(output_dir / "annotations/", exist_ok=True)
        annotation_path = output_dir / "annotations/primitives_train.json"
        offset = 9000
        anno_id = 0
        for image_id in range(offset, offset + num_val_samples):
            name = f"synthetic_diagram_{image_id}"
            width = int(
                max(min(random.gauss(DEFAULT_WIDTH, 100), MAX_WIDTH), MIN_WIDTH)
            )
            height = int(
                max(min(random.gauss(DEFAULT_HEIGHT, 100), MAX_HEIGHT), MIN_HEIGHT)
            )
            synthetic_diagram = SyntheticDiagram(
                img_size=img_size,
                width=width,
                height=height,
            )
            annotation, anno_id = synthetic_diagram.get_annotation_processed(
                name, image_id, anno_id
            )
            image_info = {
                "file_name": f"{name}.png",
                "height": height,
                "width": width,
                "id": image_id,
            }
            for primitive_annotation in annotation:
                anno_dict["annotations"].append(primitive_annotation)
            anno_dict["images"].append(image_info)
            synthetic_diagram.save_as_val(name, output_dir)
        # return annotations
        with open(annotation_path, "w") as json_file:
            json.dump(anno_dict, json_file)

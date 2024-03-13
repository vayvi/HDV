import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path
import cv2
from numpy.random import uniform, choice
from random import randint
from .helper.resources import DATABASE, BACKGROUND_RESRC_NAME
from .helper.seed import use_seed
from .element import AbstractElement

CONTEXT_BACKGROUND_COLOR = (0, 0, 0)
BACKGROUND_COLOR = (50, 50, 50)
BACKGROUND_BLUR_RADIUS_RANGE = (0, 0.2)
BACKGROUND_COLOR_BLEND_FREQ = 0.1


class BackgroundElement(AbstractElement):
    color = BACKGROUND_COLOR
    name = "background"

    @use_seed()
    def generate_content(self):
        self.img_path = self.parameters.get("image_path") or choice(
            DATABASE[BACKGROUND_RESRC_NAME]
        )
        self.img = (
            Image.open(self.img_path)
            .resize(self.size, resample=Image.LANCZOS)
            .convert("RGB")
        )
        self.blur_radius = uniform(*BACKGROUND_BLUR_RADIUS_RANGE)
        self.content_width, self.content_height = self.size
        self.pos_x, self.pos_y = (0, 0)

        color_blend = choice(
            [True, False],
            p=[BACKGROUND_COLOR_BLEND_FREQ, 1 - BACKGROUND_COLOR_BLEND_FREQ],
        )
        if color_blend:
            new_img = cv2.cvtColor(np.array(self.img), cv2.COLOR_RGB2HSV)
            new_img[:, :, 0] = randint(0, 360)
            self.img = Image.fromarray(cv2.cvtColor(new_img, cv2.COLOR_HSV2RGB))

    def to_image(self, flip=False):
        if flip:
            return self.img.transpose(Image.FLIP_LEFT_RIGHT).filter(
                ImageFilter.GaussianBlur(self.blur_radius)
            )
        else:
            return self.img.filter(ImageFilter.GaussianBlur(self.blur_radius))

    def to_label_as_array(self):
        return np.full(self.size, self.label, dtype=np.float64).transpose()

    @property
    def inherent_left_margin(self):
        img_path = (
            Path(self.img_path) if isinstance(self.img_path, str) else self.img_path
        )
        try:
            return int(
                int(img_path.parent.name) * self.width / 596
            )  # XXX: margins were calibrated on 596x842 images
        except ValueError:
            return 0

import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageFilter
from abc import ABCMeta, abstractmethod
from .helper.resources import GLYPH_FONT_RESRC_NAME, DATABASE
import string
from .helper.seed import use_seed

from numpy.random import uniform, choice
from random import randint, choice as rand_choice


class AbstractElement:
    """Abstract class that defines the characteristics of a document's element."""

    __metaclass__ = ABCMeta

    label = NotImplemented
    color = NotImplemented
    content_width = NotImplemented
    content_height = NotImplemented
    name = NotImplemented
    pos_x = NotImplemented
    pos_y = NotImplemented

    def __init__(self, width, height, seed=None, **kwargs):
        self.width, self.height = width, height
        self.parameters = kwargs
        self.generate_content()

    @property
    def size(self):
        return (self.width, self.height)

    @property
    def content_size(self):
        return (self.content_width, self.content_height)

    @property
    def position(self):
        return (self.pos_x, self.pos_y)

    @abstractmethod
    def generate_content(self):
        pass

    @abstractmethod
    def to_image(self):
        pass

    def to_image_as_array(self):
        return np.array(self.to_image(), dtype=np.float32) / 255

    @abstractmethod
    def to_label_as_array(self):
        pass

    def to_label_as_img(self):
        arr = self.to_label_as_array()
        res = np.zeros(arr.shape + (3,), dtype=np.uint8)
        res[arr == self.label] = self.color
        return Image.fromarray(res)


NEG_ELEMENT_BLUR_RADIUS_RANGE = (1, 2.5)
GLYPH_COLORED_FREQ = 0.5

POS_ELEMENT_OPACITY_RANGE = {
    "drawing": (200, 255),
    "glyph": (150, 255),
    "image": (150, 255),
    "table": (200, 255),
    "line": (0, 130),
    "table_word": (50, 200),
    "text": (200, 255),
    "diagram": (200, 255),
}

NEG_ELEMENT_OPACITY_RANGE = {
    "drawing": (0, 10),
    "glyph": (0, 10),
    "image": (0, 25),
    "table": (0, 25),
    "text": (0, 10),
    "diagram": (0, 25),
}


class GlyphElement(AbstractElement):
    font_size_range = (50, 200)
    name = "glyph"

    @use_seed()
    def generate_content(self):
        self.font_path = choice(DATABASE[GLYPH_FONT_RESRC_NAME])
        self.letter = self.parameters.get("letter") or rand_choice(
            string.ascii_uppercase
        )

        # To avoid oversized letters
        rescaled_height = (self.height * 2) // 3
        min_fs, max_fs = self.font_size_range
        actual_max_fs = min(rescaled_height, max_fs)
        tmp_font = ImageFont.truetype(self.font_path, size=actual_max_fs)
        tmp_font_width = (
            tmp_font.getbbox(self.letter)[2] - tmp_font.getbbox(self.letter)[0]
        )
        while tmp_font_width > self.width and actual_max_fs > self.font_size_range[0]:
            actual_max_fs -= 1
            tmp_font = ImageFont.truetype(self.font_path, size=actual_max_fs)
            tmp_font_width = (
                tmp_font.getbbox(self.letter)[2] - tmp_font.getbbox(self.letter)[0]
            )
        if min_fs < actual_max_fs:
            self.font_size = randint(min_fs, actual_max_fs)
        else:
            self.font_size = actual_max_fs

        self.font = ImageFont.truetype(self.font_path, size=self.font_size)
        self.as_negative = self.parameters.get("as_negative", False)
        self.blur_radius = (
            uniform(*NEG_ELEMENT_BLUR_RADIUS_RANGE) if self.as_negative else None
        )
        self.opacity = randint(
            *(
                NEG_ELEMENT_OPACITY_RANGE[self.name]
                if self.as_negative
                else POS_ELEMENT_OPACITY_RANGE[self.name]
            )
        )
        self.colored = choice(
            [True, False], p=[GLYPH_COLORED_FREQ, 1 - GLYPH_COLORED_FREQ]
        )
        self.colors = (
            (0, 0, 0)
            if not self.colored
            else tuple([randint(0, 150) for _ in range(3)])
        )
        left, upper, right, lower = self.font.getbbox(self.letter)
        self.content_width, self.content_height = right - left, lower - upper
        self.pos_x = randint(0, max(0, self.width - self.content_width))
        self.pos_y = randint(0, max(0, self.height - self.content_height))

    def to_image(self):
        canvas = Image.new("RGBA", self.size)
        image_draw = ImageDraw.Draw(canvas)
        colors_alpha = self.colors + (self.opacity,)
        image_draw.text(self.position, self.letter, font=self.font, fill=colors_alpha)
        if self.as_negative:
            canvas = canvas.filter(ImageFilter.GaussianBlur(self.blur_radius))
        return canvas

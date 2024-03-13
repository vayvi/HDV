from pathlib import Path
from synthetic_module.helper.seed import use_seed
from PIL import Image
from random import randint
from synthetic_module.helper.resources import DATABASE, NOISE_PATTERN_RESRC_NAME
from numpy.random import uniform, choice
from random import randint
from synthetic_module.helper.image import resize
import numpy as np

NOISE_PATTERN_SIZE_RANGE = {
    "border_hole": (100, 600),
    "center_hole": (100, 400),
    "corner_hole": (100, 400),
    "phantom_character": (30, 100),
}
NOISE_PATTERN_OPACITY_RANGE = (0.2, 0.6)


@use_seed()
def get_random_noise_pattern(width, height):
    pattern_path = choice(DATABASE[NOISE_PATTERN_RESRC_NAME])
    pattern_type = Path(pattern_path).parent.name
    img = Image.open(pattern_path).convert("L")
    size_min, size_max = NOISE_PATTERN_SIZE_RANGE[pattern_type]
    size_max = min(min(width, height), size_max)
    size = (randint(size_min, size_max), randint(size_min, size_max))
    if pattern_type in ["border_hole", "corner_hole"]:
        img = resize(img, size, keep_aspect_ratio=True, resample=Image.LANCZOS)
        rotation = choice([None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270])
        if rotation is not None:
            img = img.transpose(rotation)
        if pattern_type == "border_hole":
            if rotation is None:
                position = (randint(0, width - img.size[0]), 0)
            elif rotation == Image.ROTATE_90:
                position = (0, randint(0, height - img.size[1]))
            elif rotation == Image.ROTATE_180:
                position = (randint(0, width - img.size[0]), height - img.size[1])
            else:
                position = (width - img.size[0], randint(0, height - img.size[1]))
        else:
            if rotation is None:
                position = (0, 0)
            elif rotation == Image.ROTATE_90:
                position = (0, height - img.size[1])
            elif rotation == Image.ROTATE_180:
                position = (width - img.size[0], height - img.size[1])
            else:
                position = (width - img.size[0], 0)
    else:
        img = resize(img, size, keep_aspect_ratio=False, resample=Image.LANCZOS)
        rotation = randint(0, 360)
        img = img.rotate(rotation, fillcolor=255)
        pad = max(img.width, img.height)
        position = (randint(0, max(0, width - pad)), randint(0, max(0, height - pad)))

    alpha = uniform(*NOISE_PATTERN_OPACITY_RANGE)
    arr = np.array(img.convert("RGBA"))
    arr[:, :, 3] = (255 - arr[:, :, 2]) * alpha
    hue_color = randint(0, 360)
    value_ratio = uniform(0.95, 1)
    return Image.fromarray(arr), hue_color, value_ratio, position

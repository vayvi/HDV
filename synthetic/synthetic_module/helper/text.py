from synthetic_module.helper.seed import use_seed
from numpy.random import choice
from synthetic_module.helper.resources import (
    DATABASE,
    TEXT_RESRC_NAME,
    FONT_RESRC_NAME,
    CHINESE_TEXT_RESRC_NAME,
    ARABIC_TEXT_RESRC_NAME,
)
from random import randint
from PIL import ImageFont

# FONT_TYPES = ["arabic", "chinese", "handwritten", "normal"]
FONT_TYPES = ["handwritten", "normal"]
# TEXT_FONT_TYPE_RATIO = {
#     "arabic": 0.25,
#     "chinese": 0.25,
#     "handwritten": 0.3,
#     "normal": 0.2,
# }
TEXT_FONT_TYPE_RATIO = {
    "handwritten": 0.5,
    "normal": 0.5,
}
MIN_NB_CHARACTERS = 100
MIN_IMG_DIMENSION = 200
TEXT_BASELINE_HEIGHT = 5
TEXT_BBOX_FREQ = 0

TEXT_BBOX_BORDER_WIDTH_RANGE = (1, 6)
TEXT_BBOX_PADDING_RANGE = (0, 20)
TEXT_COLORED_FREQ = 0.5
TEXT_JUSTIFIED_PARAGRAPH_FREQ = 0.7
TEXT_ROTATION_ANGLE_RANGE = (-60, 60)
TEXT_TIGHT_PARAGRAPH_FREQ = 0.5
TEXT_TITLE_UPPERCASE_RATIO = 0.5
TEXT_TITLE_UNILINE_RATIO = 0.25
TEXT_UNDERLINED_FREQ = 0


TEXT_UNDERLINED_PADDING_RANGE = (0, 4)
FONT_SIZE_RANGE = (15, 40)


@use_seed()
def get_random_font():
    font_type = choice(
        list(TEXT_FONT_TYPE_RATIO.keys()), p=list(TEXT_FONT_TYPE_RATIO.values())
    )
    return choice(DATABASE[FONT_RESRC_NAME][font_type])


def get_dictionary(parameters, height):
    min_fs, max_fs = FONT_SIZE_RANGE
    font_path = parameters.get("font_path") or get_random_font()

    rescaled_height = (height * 2) // 3  # to avoid oversized letters
    actual_max_fs = min(rescaled_height, max_fs)
    if min_fs < actual_max_fs:
        font_size = randint(min_fs, actual_max_fs)
    else:
        font_size = actual_max_fs
    font = ImageFont.truetype(font_path, size=font_size)
    if "text" in parameters:
        text = parameters["text"]
    else:
        n_char = 0
        if "chinese" in font_path:
            text_resource = DATABASE[CHINESE_TEXT_RESRC_NAME]
        elif "arabic" in font_path:
            text_resource = DATABASE[ARABIC_TEXT_RESRC_NAME]
        else:
            text_resource = DATABASE[TEXT_RESRC_NAME]
        while n_char <= 100:
            text_path = choice(text_resource)
            with open(text_path) as f:
                text = f.read().rstrip("\n")
            n_char = len(text)
    if "chinese" in font_path:
        dictionary = list(text)
    else:
        dictionary = text.split(" ")
    return dictionary, font

from synthetic_module.helper.path import (
    coerce_to_path_and_check_exist,
    get_files_from_dir,
)
from synthetic_module import SYNTHETIC_RESRC_PATH


# resource names
BACKGROUND_RESRC_NAME = "background"
FONT_RESRC_NAME = "font"
FONT_TYPES = ["arabic", "chinese", "handwritten", "normal"]
NOISE_PATTERN_RESRC_NAME = "noise_pattern"
TEXT_RESRC_NAME = "text"
ARABIC_TEXT_RESRC_NAME = "text_arabic"
CHINESE_TEXT_RESRC_NAME = "text_chinese"
GLYPH_FONT_RESRC_NAME = "glyph_font"

AVAILABLE_RESRC_NAMES = [
    BACKGROUND_RESRC_NAME,
    FONT_RESRC_NAME,
    NOISE_PATTERN_RESRC_NAME,
    TEXT_RESRC_NAME,
    ARABIC_TEXT_RESRC_NAME,
    CHINESE_TEXT_RESRC_NAME,
    GLYPH_FONT_RESRC_NAME,
]
VALID_EXTENSIONS = {
    BACKGROUND_RESRC_NAME: ["jpeg", "jpg", "png"],
    FONT_RESRC_NAME: ["otf", "ttf"],
    NOISE_PATTERN_RESRC_NAME: ["png"],
    GLYPH_FONT_RESRC_NAME: ["otf", "ttf"],
    TEXT_RESRC_NAME: ["txt"],
    ARABIC_TEXT_RESRC_NAME: ["txt"],
    CHINESE_TEXT_RESRC_NAME: ["txt"],
}

# ResourceDownloader constants
DEFAULT_LANGUAGES = ["en"]


class ResourceDatabase:
    def __init__(self, input_dir=SYNTHETIC_RESRC_PATH):
        self.input_dir = coerce_to_path_and_check_exist(input_dir)
        self.table = self._initialize_table()

    def _initialize_table(self):
        table = dict()
        for name in AVAILABLE_RESRC_NAMES:
            p, ext = self.input_dir / name, VALID_EXTENSIONS[name]
            if name == FONT_RESRC_NAME:
                d = {}
                for font in FONT_TYPES:
                    files = get_files_from_dir(
                        p / font, valid_extensions=ext, recursive=True
                    )
                    d[font] = list(map(str, files))
                table[name] = d
            else:
                files = get_files_from_dir(p, valid_extensions=ext, recursive=True)
                table[name] = list(map(str, files))
        return table

    @property
    def name(self):
        return self.input_dir.name

    @property
    def resource_names(self):
        return list(self.table.keys())

    def __getitem__(self, key):
        return self.table[key]

    def __repr__(self):
        return '{}("{}")'.format(self.__class__.__name__, self.name)

    def __str__(self):
        return "\n\t".join([self.name] + self.resource_names)


DATABASE = ResourceDatabase()

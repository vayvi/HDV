from PIL import Image, ImageDraw, ImageFilter
import numpy as np


def resize(img, size, keep_aspect_ratio=True, resample=Image.LANCZOS):
    if isinstance(size, (int, float)):
        assert keep_aspect_ratio
        ratio = float(np.sqrt(size / (img.size[0] * img.size[1])))
        size = round(ratio * img.size[0]), round(ratio * img.size[1])
    elif keep_aspect_ratio:
        ratio = float(
            min([s1 / s2 for s1, s2 in zip(size, img.size)])
        )  # XXX bug with np.float64 and round
        size = round(ratio * img.size[0]), round(ratio * img.size[1])

    return img.resize(size, resample=resample)


def paste_with_blured_borders(canvas, img, position=(0, 0), border_width=3):
    # TODO: understand this
    canvas.paste(img, position)
    mask = Image.new("L", canvas.size, 0)
    draw = ImageDraw.Draw(mask)
    x0, y0 = [position[k] - border_width for k in range(2)]
    x1, y1 = [position[k] + img.size[k] + border_width for k in range(2)]

    diam = 2 * border_width
    for d in range(diam + border_width):
        x1, y1 = x1 - 1, y1 - 1
        alpha = 255 if d < border_width else int(255 * (diam + border_width - d) / diam)
        fill = None if d != diam + border_width - 1 else 0
        draw.rectangle([x0, y0, x1, y1], fill=fill, outline=alpha)
        x0, y0 = x0 + 1, y0 + 1

    blur = canvas.filter(ImageFilter.GaussianBlur(border_width / 2))
    canvas.paste(blur, mask=mask)

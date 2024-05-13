from gettext import find
from hmac import new
from numpy.random import uniform, choice
from random import randint, choice as rand_choice
import numpy as np
from sympy import rad

from .helper.seed import use_seed
from .helper.text import get_dictionary
from .element import AbstractElement
from PIL import Image, ImageDraw, ImageFilter

POS_ELEMENT_OPACITY_RANGE = {
    "drawing": (220, 255),
    "glyph": (150, 255),
    "image": (150, 255),
    "table": (200, 255),
    "line": (120, 200),
    "table_word": (50, 200),
    "text": (200, 255),
    "diagram": (180, 255),
}

NEG_ELEMENT_OPACITY_RANGE = {
    "drawing": (0, 10),
    "glyph": (0, 10),
    "image": (0, 25),
    "table": (0, 25),
    "text": (0, 10),
    "diagram": (0, 25),
}
NEG_ELEMENT_BLUR_RADIUS_RANGE = (1, 2.5)
WIDTH_VALUES = [1, 2, 3, 4]

DIAGRAM_COLOR = (255, 100, 180)

COCENTRIC_CIRCLES_RATIO = 0.2
SAME_RADIUS_CIRCLES_RATIO = 0.1
COLORED_FREQ = 0.8


# NOTE identical from src/util/primitives.py find_circle_center()
def find_circle_center(p1, p2, p3):
    """Circle center from 3 points"""
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
    if abs(det) < 1.0e-10:
        return (None, None)

    cx = (bc * (p2[1] - p3[1]) - cd * (p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
    return np.array([cx, cy])


# NOTE identical from src/util/primitives.py get_angles_from_arc_points()
def get_angles_from_arc_points(p0, p_mid, p1):
    arc_center = find_circle_center(p0, p_mid, p1)
    arc_center = (arc_center[0], arc_center[1])
    start_angle = np.arctan2(p0[1] - arc_center[1], p0[0] - arc_center[0])
    end_angle = np.arctan2(p1[1] - arc_center[1], p1[0] - arc_center[0])
    mid_angle = np.arctan2(p_mid[1] - arc_center[1], p_mid[0] - arc_center[0])
    return start_angle, mid_angle, end_angle, arc_center


def find_circle_center_arr(p1, p2, p3):
    """Circle center from 3 points"""
    temp = p2[:, 0] ** 2 + p2[:, 1] ** 2
    bc = (p1[:, 0] ** 2 + p1[:, 1] ** 2 - temp) / 2
    cd = (temp - p3[:, 0] ** 2 - p3[:, 1] ** 2) / 2
    det = (p1[:, 0] - p2[:, 0]) * (p2[:, 1] - p3[:, 1]) - (p2[:, 0] - p3[:, 0]) * (
        p1[:, 1] - p2[:, 1]
    )

    # Handle the case where the determinant is close to zero
    mask = np.abs(det) < 1.0e-10
    det[mask] = 1.0  # Prevent division by zero
    bc[mask] = 0.0  # These arcs will have center at (0, 0)
    cd[mask] = 0.0

    cx = (bc * (p2[:, 1] - p3[:, 1]) - cd * (p1[:, 1] - p2[:, 1])) / det
    cy = ((p1[:, 0] - p2[:, 0]) * cd - (p2[:, 0] - p3[:, 0]) * bc) / det
    return np.stack([cx, cy], axis=-1)


def get_angles_from_arc_points_arr(p0, p_mid, p1):
    arc_center = find_circle_center_arr(p0, p_mid, p1)
    start_angle = np.arctan2(p0[:, 1] - arc_center[:, 1], p0[:, 0] - arc_center[:, 0])
    end_angle = np.arctan2(p1[:, 1] - arc_center[:, 1], p1[:, 0] - arc_center[:, 0])
    mid_angle = np.arctan2(
        p_mid[:, 1] - arc_center[:, 1], p_mid[:, 0] - arc_center[:, 0]
    )
    to_deg = lambda x: (x * 180 / np.pi) % 360
    start_angle = to_deg(start_angle)
    end_angle = to_deg(end_angle)
    mid_angle = to_deg(mid_angle)
    return start_angle, mid_angle, end_angle, arc_center


def gen_arc_from_p0_p1_radius(p0, p1, radius):
    p0, p1 = np.array(p0, dtype=np.float64), np.array(p1, dtype=np.float64)
    midpoint = (p0 + p1) / 2
    diff = p1 - p0
    dist = np.linalg.norm(diff)
    if dist != 0:
        unit_diff = diff / dist
        unit_perpendicular = np.array([-unit_diff[1], unit_diff[0]])
        offset = unit_perpendicular * np.sqrt(radius**2 - (dist / 2) ** 2)
    else:
        offset = np.array([0, radius])
    arc_center = midpoint + offset
    start_angle = np.arctan2(p0[1] - arc_center[1], p0[0] - arc_center[0])
    end_angle = np.arctan2(p1[1] - arc_center[1], p1[0] - arc_center[0])
    return start_angle, end_angle, arc_center


def is_valid_arc(
    arc_center,
    radius,
    start_angle,
    end_angle,
    width,
    height,
    min_angle=20,
    threshold_dist=5,
):
    angle1, angle2 = start_angle, end_angle
    if angle1 > angle2:
        angle2 += 2 * np.pi
    if np.abs(start_angle - end_angle) * 180 / np.pi % 360 < min_angle:
        return False

    testing_angles = np.linspace(angle1, angle2, 30)
    testing_pts = arc_center.reshape(2, 1) + radius * np.array(
        [np.cos(testing_angles), np.sin(testing_angles)]
    )
    if (
        (testing_pts[0, :] < 0).any()
        or (testing_pts[0, :] > width).any()
        or (testing_pts[1, :] < 0).any()
        or (testing_pts[1, :] > height).any()
    ):
        return False

    if np.linalg.norm(testing_pts[:, 0] - testing_pts[:, -1]) < threshold_dist:
        return False
    return True


class DiagramElement(AbstractElement):
    color = DIAGRAM_COLOR
    name = "diagram"

    @use_seed()
    def generate_content(self):
        dictionary, self.font = get_dictionary(self.parameters, self.height)
        self.diagram_position = self.parameters["diagram_position"]
        self.as_negative = self.parameters.get("as_negative", False)
        self.thickness_range = self.parameters.get("thickness_range", WIDTH_VALUES)
        self.blur_radius = (
            uniform(*NEG_ELEMENT_BLUR_RADIUS_RANGE) if self.as_negative else None
        )
        # self.opacity = randint(
        #     *NEG_ELEMENT_OPACITY_RANGE[self.name]
        #     if self.as_negative
        #     else POS_ELEMENT_OPACITY_RANGE[self.name]
        # )
        self.colored = choice([True, False], p=[COLORED_FREQ, 1 - COLORED_FREQ])
        self.colors = (
            tuple([randint(0, 60)] * 3)
            if not self.colored
            else tuple([randint(0, 255) for _ in range(3)])
        )
        self.threshold_dist = self.parameters.get("threshold_dist", 20)
        # self.colors_alpha = self.colors + (self.opacity,)
        self.number_circles = self.parameters.get(
            "number_circles", int(max((np.random.normal(randint(2, 10), 2)), 1))
        )
        self.number_arcs = self.parameters.get(
            "number_arcs", int(max((np.random.normal(randint(2, 20), 2)), 1))
        )
        self.number_lines = self.parameters.get(
            "number_lines", int(max((np.random.normal(randint(2, 20), 2)), 1))
        )
        self.number_words = self.parameters.get(
            "number_words", int(max((np.random.normal(randint(2, 20), 2)), 0))
        )
        # self.number_circles = self.number_lines = 0
        self.flower_arcs = self.parameters.get(
            "flower_arcs", choice([True, False], p=[0.1, 0.9])
        )

        self.same_color = choice([True, False], p=[0.8, 0.2])
        # # For mock diagram
        # self.number_lines = 6
        # self.number_arcs = 4
        # self.number_circles = 2
        # self.flower_arcs = True
        # self.same_color = True
        # self.colored = False
        # self.colors = ((0,0,0))

        self.content_width = self.parameters.get("content_width", None)
        self.content_height = self.parameters.get("content_height", None)
        self.fill = choice([False, True], p=[0.8, 0.2])
        self.table, self.content_width, self.content_height = self._generate_diagram(
            dictionary, self.content_width, self.content_height
        )
        self.pos_x = randint(self.diagram_position[0], self.width - self.content_width)
        self.pos_y = randint(
            self.diagram_position[1], self.height - self.content_height
        )

    @use_seed()
    def _generate_diagram(self, dictionary, width=None, height=None):
        if width is None:
            width = randint(
                max(self.diagram_position[0], self.width // 3),
                self.width - 2 * self.diagram_position[0],
            )
        if height is None:
            height = randint(
                max(self.diagram_position[1], self.height // 3),
                self.height - 2 * self.diagram_position[1],
            )
        to_deg = lambda x: (x * 180 / np.pi) % 360
        circle_pos, circle_radius = [], []
        for i in range(self.number_circles):
            radius = np.random.uniform(
                min(10, min(width, height) // 2.1), min(width, height) // 2.1
            )
            center_x = np.random.uniform(radius, width - radius)
            center_y = np.random.uniform(radius, height - radius)
            circle_radius.append(radius)
            circle_pos.append((center_x, center_y))
            add_cocentric_circles = choice(
                [True, False], p=[COCENTRIC_CIRCLES_RATIO, 1 - COCENTRIC_CIRCLES_RATIO]
            )
            if add_cocentric_circles and radius > (min(width, height) // 6):
                num_circles_2 = 2 * randint(2, 8)

                for k in range(0, num_circles_2, 3):
                    new_radius = radius * np.random.uniform(
                        (k + 1) / num_circles_2, min((k + 2) / num_circles_2, 1)
                    )
                    circle_radius.append(new_radius)
                    circle_pos.append((center_x, center_y))

            add_same_radius_circles = choice(
                [True, False],
                p=[SAME_RADIUS_CIRCLES_RATIO, 1 - SAME_RADIUS_CIRCLES_RATIO],
            )
            if add_same_radius_circles:
                num_circles_2 = 2 * randint(2, 10)

                for k in range(0, num_circles_2, 2):
                    # skip = choice([True, False], p=[0.2, 0.8])
                    # if skip:
                    #     continue
                    new_angle = np.random.uniform(
                        2 * np.pi * (k + 1) / num_circles_2,
                        2 * (k + 2) * np.pi / num_circles_2,
                    )
                    new_center_x = np.clip(
                        center_x + radius * np.cos(new_angle), radius, width - radius
                    )
                    new_center_y = np.clip(
                        center_y + radius * np.sin(new_angle), radius, height - radius
                    )

                    circle_radius.append(radius)
                    circle_pos.append((new_center_x, new_center_y))

        arc_centers, arc_radius, arc_angles = [], [], []
        to_deg = lambda x: (x * 180 / np.pi) % 360
        line_coords = []
        min_border = min(width, height)

        for i in range(self.number_arcs):
            shared_endpoints_arcs = choice([True, False], p=[0.2, 0.8])
            shared_point_arcs = choice([True, False], p=[0.2, 0.8])
            horizontal_arc = choice([True, False], p=[0.2, 0.8])
            vertical_arc = choice(
                [True, False], p=[0.25, 0.75]
            )  # FIXME fix the distribution
            p0 = np.array(
                [np.random.uniform(10, width - 10), np.random.uniform(10, height - 10)]
            )
            if horizontal_arc:
                p1 = np.array([np.random.randint(10, width - 10), p0[1]])
            elif vertical_arc:
                p1 = np.array([p0[0], np.random.uniform(10, height - 10)])

            p1 = np.array(
                [np.random.uniform(10, width - 10), np.random.uniform(10, height - 10)]
            )
            try:
                radius = np.random.uniform(
                    max(
                        min_border // 20, np.linalg.norm((p1 - p0)) // 1.9 + 10
                    ),  # cant have a radius smaller than the mid distance between the two points
                    min_border * 2,
                )
            except ValueError as e:
                print(e)
                radius = np.linalg.norm((p1 - p0)) // 1.5

            start_angle, end_angle, arc_center = gen_arc_from_p0_p1_radius(
                p0, p1, radius
            )
            valid_arc = is_valid_arc(
                arc_center, radius, start_angle, end_angle, width, height
            )
            if not valid_arc:
                continue

            start_angle, end_angle = to_deg(start_angle), to_deg(end_angle)

            arc_radius.append(radius)
            arc_centers.append(arc_center)
            arc_angles.append((start_angle, end_angle))
            if shared_endpoints_arcs:
                num_arcs = randint(2, 8)
                new_radius = radius
                for k in range(num_arcs):
                    if uniform() < 0.5:
                        new_p0, new_p1 = p1, p0
                    else:
                        new_p0, new_p1 = p0, p1
                    # if uniform() < 0.5:
                    #     new_p0 = np.array(
                    #         [np.random.randint(0, width), np.random.randint(0, height)]
                    #     )
                    new_radius = new_radius * np.random.uniform(1.5, 3)

                    (
                        new_start_angle,
                        new_end_angle,
                        new_arc_center,
                    ) = gen_arc_from_p0_p1_radius(new_p0, new_p1, new_radius)
                    valid_arc = is_valid_arc(
                        new_arc_center,
                        new_radius,
                        new_start_angle,
                        new_end_angle,
                        width,
                        height,
                    )
                    if not valid_arc:
                        break

                    new_start_angle, new_end_angle = to_deg(new_start_angle), to_deg(
                        new_end_angle
                    )
                    arc_centers.append(new_arc_center)
                    arc_radius.append(new_radius)
                    arc_angles.append((new_start_angle, new_end_angle))
            if shared_point_arcs:
                num_arcs = randint(2, 8)
                for k in range(num_arcs):
                    interpolation_factor = np.random.uniform(0.2, 0.8)
                    new_radius = radius * np.random.uniform(1.0, 1.4)
                    new_start_angle = (
                        interpolation_factor * start_angle * np.pi / 180
                        + (1 - interpolation_factor) * end_angle * np.pi / 180
                    )

                    new_p0 = arc_center + radius * np.array(
                        [np.cos(new_start_angle), np.sin(new_start_angle)]
                    )  # choosing point that lies on the arc
                    new_p1 = np.array(
                        [np.random.randint(0, width), np.random.randint(0, height)]
                    )
                    if uniform() < 0.5:
                        new_p0, new_p1 = new_p1, new_p0
                    new_radius = max(
                        radius, np.linalg.norm(new_p0 - new_p1) / 2 + 10
                    ) * np.random.uniform(1.0, 1.6)
                    (
                        new_start_angle,
                        new_end_angle,
                        new_arc_center,
                    ) = gen_arc_from_p0_p1_radius(new_p0, new_p1, new_radius)

                    valid_arc = is_valid_arc(
                        new_arc_center,
                        new_radius,
                        new_start_angle,
                        new_end_angle,
                        width,
                        height,
                    )
                    if not valid_arc:
                        continue
                    # line_coords.append((new_p0[0], new_p0[1], new_p1[0], new_p1[1]))
                    new_start_angle, new_end_angle = to_deg(new_start_angle), to_deg(
                        new_end_angle
                    )
                    arc_centers.append(new_arc_center)
                    arc_radius.append(new_radius)
                    arc_angles.append((new_start_angle, new_end_angle))

        if self.flower_arcs:
            num_arcs = randint(2, 10)
            radius = randint(
                min(40, min(width, height) // 2.1), min(width, height) // 2.1
            )
            center_x = np.random.randint(radius, width - radius)
            center_y = np.random.randint(radius, height - radius)
            circle_center = np.array([center_x, center_y]).reshape(1, 2)
            if choice([True, False]):  # show circle
                circle_radius.append(radius)
                circle_pos.append((center_x, center_y))
            # first_angle = choice([0, np.pi / 2, np.pi, 3 * np.pi / 2])
            start_angles = np.linspace(0, np.pi, num_arcs) + np.random.normal(
                0, 2, num_arcs
            )

            end_angles = (np.pi + start_angles) + np.random.normal(0, 2, num_arcs)

            start_pts = (
                radius * np.array([np.cos(start_angles), np.sin(start_angles)]).T
                + circle_center
            )
            end_pts = (
                radius * np.array([np.cos(end_angles), np.sin(end_angles)]).T
                + circle_center
            )
            noise_angle = np.random.uniform(0, 2 * np.pi)
            noise_distance = np.random.uniform(radius // 8, radius // 2)
            noise_distance = 0
            mid_point = circle_center + noise_distance * np.array(
                [np.cos(noise_angle), np.sin(noise_angle)]
            )
            mid_pts = np.repeat(mid_point.reshape(1, 2), num_arcs, axis=0)
            current_arc_centers = find_circle_center_arr(start_pts, mid_pts, end_pts)
            radii = np.linalg.norm(current_arc_centers - start_pts, axis=1)

            for k in range(num_arcs):
                p0, p1 = start_pts[k, :], end_pts[k, :]
                radius = radii[k]
                start_angle, end_angle, arc_center = gen_arc_from_p0_p1_radius(
                    p0, p1, radius
                )

                def is_large_arc(rad_angle):
                    if rad_angle[0] <= np.pi:
                        return not (
                            rad_angle[0] < rad_angle[1] < (np.pi + rad_angle[0])
                        )
                    return (rad_angle[0] - np.pi) < rad_angle[1] < rad_angle[0]

                if is_large_arc((start_angle, end_angle)):
                    start_angle, end_angle = end_angle, start_angle
                valid_arc = is_valid_arc(
                    arc_center, radius, start_angle, end_angle, width, height
                )
                if not valid_arc:
                    continue
                start_angle, end_angle = to_deg(start_angle), to_deg(end_angle)
                arc_centers.append(arc_center)
                arc_radius.append(radius)
                arc_angles.append((start_angle, end_angle))

        arc_centers = np.array(arc_centers)
        arc_angles = np.array(arc_angles)
        arc_radius = np.array(arc_radius)
        circle_pos = np.array(circle_pos)
        circle_radius = np.array(circle_radius)

        if len(circle_pos) > self.number_circles:
            keep_indices = np.random.choice(
                len(circle_pos), size=self.number_circles, replace=False
            )
            circle_pos = circle_pos[keep_indices]
            circle_radius = circle_radius[keep_indices]

        if len(arc_centers) > self.number_arcs:
            keep_indices = np.random.choice(
                len(arc_centers), size=self.number_arcs, replace=False
            )
            arc_centers = arc_centers[keep_indices]
            arc_radius = arc_radius[keep_indices]
            arc_angles = arc_angles[keep_indices]

        line_coords = []
        for i in range(self.number_lines):
            length = randint(10, min(width // 2 - 1, height // 2 - 1))
            coords_x = np.random.randint(length, width - length)
            coords_y = np.random.randint(length, height - length)

            angle = np.random.uniform(0, 2 * np.pi)
            x_length = length * np.abs(np.cos(angle))
            y_length = length * np.sin(angle)
            direction = choice([1, -1])
            coords = (
                coords_x - x_length,
                coords_y - direction * y_length,
                coords_x + x_length,
                coords_y + direction * y_length,
            )
            line_coords.append(coords)
        shared_point_lines = choice([True, False], p=[0.1, 0.9])
        if shared_point_lines:
            length = randint(10, min(width // 2 - 1, height // 2 - 1))
            start_x = np.random.randint(length, width - length)
            start_y = np.random.randint(length, height - length)

            for num_line in range(randint(2, 10)):
                if choice([True, False], p=[0.2, 0.8]):
                    length = np.random.uniform(0.8, 0.9) * length
                angle = np.random.uniform(0, 2 * np.pi)
                x_length = length * np.abs(np.cos(angle))
                y_length = length * np.sin(angle)
                direction = choice([1, -1])
                coords = (
                    start_x,
                    start_y,
                    start_x + x_length,
                    start_y + direction * y_length,
                )
                line_coords.append(coords)
        horizontal_lines = choice([True, False], p=[0.1, 0.9])
        vertical_lines = choice([True, False], p=[0.1, 0.9])
        if horizontal_lines:
            for num_line in range(randint(1, 10)):
                start_x = np.random.randint(0, width)
                start_y = np.random.randint(0, height)
                end_x = np.random.randint(0, width)
                end_y = start_y
                line_coords.append((start_x, start_y, end_x, end_y))
        if vertical_lines:
            for num_line in range(randint(1, 10)):
                start_x = np.random.randint(0, width)
                start_y = np.random.randint(0, height)
                end_x = start_x
                end_y = np.random.randint(0, height)
                line_coords.append((start_x, start_y, end_x, end_y))
        line_coords = np.array(line_coords)

        if len(line_coords) > self.number_lines:
            keep_indices = np.random.choice(
                len(line_coords), size=self.number_lines, replace=False
            )
            line_coords = line_coords[keep_indices]
        words, word_positions = [], []
        for i in range(self.number_words):
            word_as_number = choice([True, False], p=[0.1, 0.9])

            if word_as_number:
                n_letter = randint(1, 4)
                word = f"{randint(0, 10**n_letter - 1):,}"

            else:
                word = rand_choice(dictionary)

                uppercase = choice([True, False])
                if uppercase:
                    word = word.upper()

            if len(word) > 0:
                try:
                    left, upper, right, lower = self.font.getbbox(word)
                    w = right - left
                    h = lower - upper
                except OSError:
                    continue
                try:
                    top_left_x = np.random.randint(0, width - w)
                    top_left_y = np.random.randint(0, height - h)
                except ValueError:
                    # print(
                    #     f"Word {word} is too long for the diagram. {w} > {width} or {h} > {height}"
                    # )
                    continue
                words.append(word)
                word_positions.append((top_left_x, top_left_y))

        return (
            {
                "circle_pos": circle_pos,
                "circle_radius": circle_radius,
                "line_coords": line_coords,
                "arc_centers": arc_centers,
                "arc_radius": arc_radius,
                "arc_angles": arc_angles,
                "words": words,
                "word_positions": word_positions,
            },
            width,
            height,
        )

    @use_seed()
    def to_image(self):
        canvas = Image.new("RGBA", self.size)
        draw = ImageDraw.Draw(canvas)
        # TODO: either fill the first circle alone, or fill cocentric circles only
        # only filling the first circle for now
        if self.fill:
            opacity = randint(40, 80)
            fill_color = tuple(randint(0, 255) for _ in range(3)) + (opacity,)
        else:
            fill_color = None
        prev_circle_radius = 0
        prev_circle_pos = np.array([0, 0])
        for circle_pos, circle_radius in zip(
            self.table["circle_pos"], self.table["circle_radius"]
        ):
            # opacity = randint(*NEG_ELEMENT_OPACITY_RANGE[self.name])
            if self.same_color:
                colors = self.colors
            else:
                colors = (
                    tuple([randint(0, 60)] * 3)
                    if not self.colored
                    else tuple([randint(0, 255) for _ in range(3)])
                )
            # opacity = 255
            keep_same_params = (prev_circle_radius == circle_radius) or (
                prev_circle_pos == circle_pos
            ).all()
            if not keep_same_params:
                params = {
                    "fill": fill_color,
                    "outline": colors,
                    "width": rand_choice(self.thickness_range),
                }

            center = [self.pos_x + circle_pos[0], self.pos_y + circle_pos[1]]
            shape = [
                center[0] - circle_radius,
                center[1] - circle_radius,
                center[0] + circle_radius,
                center[1] + circle_radius,
            ]
            fill_color = None  # only fill the first circle to not overlap
            draw.ellipse(shape, **params)
            prev_circle_radius = circle_radius
            prev_circle_pos = circle_pos
        for arc_center, arc_radius, arc_angles in zip(
            self.table["arc_centers"],
            self.table["arc_radius"],
            self.table["arc_angles"],
        ):
            # opacity = randint(*NEG_ELEMENT_OPACITY_RANGE[self.name])
            if self.same_color:
                colors = self.colors
            else:
                colors = (
                    tuple([randint(0, 60)] * 3)
                    if not self.colored
                    else tuple([randint(0, 255) for _ in range(3)])
                )
            params = {
                "fill": colors,
                "width": rand_choice(self.thickness_range),
            }

            center = [self.pos_x + arc_center[0], self.pos_y + arc_center[1]]
            shape = [
                center[0] - arc_radius,
                center[1] - arc_radius,
                center[0] + arc_radius,
                center[1] + arc_radius,
            ]

            draw.arc(shape, start=arc_angles[0], end=arc_angles[1], **params)

        for line_coords in self.table["line_coords"]:
            # opacity = randint(*POS_ELEMENT_OPACITY_RANGE[self.name])
            if self.same_color:
                colors = self.colors
            else:
                colors = (
                    tuple([randint(0, 60)] * 3)
                    if not self.colored
                    else tuple([randint(0, 255) for _ in range(3)])
                )
            params = {
                "fill": colors,
                "width": rand_choice(self.thickness_range),
            }

            draw.line(
                [
                    self.pos_x + line_coords[0],
                    self.pos_y + line_coords[1],
                    self.pos_x + line_coords[2],
                    self.pos_y + line_coords[3],
                ],
                **params,
            )
            assert self.pos_x + line_coords[0] < self.width, f"{line_coords}"
            assert self.pos_y + line_coords[1] < self.height, f"{line_coords}"
            assert self.pos_x + line_coords[2] < self.width, f"{line_coords}"
            assert self.pos_y + line_coords[3] < self.height, f"{line_coords}"

        for word, pos in zip(self.table["words"], self.table["word_positions"]):
            opacity = randint(*POS_ELEMENT_OPACITY_RANGE[self.name])

            # colors_alpha = self.colors + (opacity,)
            if self.same_color:
                colors = self.colors
            else:
                colors = (
                    tuple([randint(0, 60)] * 3)
                    if not self.colored
                    else tuple([randint(0, 255) for _ in range(3)])
                )
            pos = pos[0] + self.pos_x, pos[1] + self.pos_y

            draw.text(pos, word, font=self.font, fill=colors)

        if self.as_negative:
            canvas = canvas.filter(ImageFilter.GaussianBlur(self.blur_radius))

        return canvas

    def get_annotation(self):
        centers, circle_radii, lines, arcs = [], [], [], []
        self.offset = [self.pos_x, self.pos_y]
        for circle_pos, circle_radius in zip(
            self.table["circle_pos"], self.table["circle_radius"]
        ):
            center = [self.offset[0] + circle_pos[0], self.offset[1] + circle_pos[1]]
            centers.append(center)
            circle_radii.append(circle_radius)
        for line_coords in self.table["line_coords"]:
            lines.append(
                [
                    int(self.offset[0] + line_coords[0]),
                    int(self.offset[1] + line_coords[1]),
                    int(self.offset[0] + line_coords[2]),
                    int(self.offset[1] + line_coords[3]),
                ]
            )
        for arc_center, arc_radius, arc_angles in zip(
            self.table["arc_centers"],
            self.table["arc_radius"],
            self.table["arc_angles"],
        ):
            rad_angle = arc_angles * np.pi / 180
            start_angle, end_angle = rad_angle

            p0 = arc_radius * np.array(
                [
                    np.cos(start_angle),
                    np.sin(start_angle),
                ]
            ) + np.array(arc_center)
            p1 = arc_radius * np.array(
                [
                    np.cos(end_angle),
                    np.sin(end_angle),
                ]
            ) + np.array(arc_center)
            if start_angle > end_angle:
                end_angle += 2 * np.pi

            mid_angle = (start_angle + end_angle) / 2
            mid_angle, end_angle = mid_angle % (2 * np.pi), end_angle % (2 * np.pi)

            p_mid = arc_center + arc_radius * np.array(
                [np.cos(mid_angle), np.sin(mid_angle)]
            )
            arcs.append(
                [
                    int(self.offset[0] + p0[0]),
                    int(self.offset[1] + p0[1]),
                    int(self.offset[0] + p1[0]),
                    int(self.offset[1] + p1[1]),
                    int(self.offset[0] + p_mid[0]),
                    int(self.offset[1] + p_mid[1]),
                ]
            )

        return {
            "circle_centers": centers,
            "circle_radii": circle_radii,
            "lines": lines,
            "arcs": arcs,
        }

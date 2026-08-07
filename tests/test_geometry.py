import cv2
import numpy as np
import pytest

from baas_student_recognition_ai.geometry import FixedLessonLayout


@pytest.mark.parametrize("count", range(1, 10))
def test_fixed_layout_supports_one_to_n_nonempty_cards(count):
    layout = FixedLessonLayout()
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    for index in range(count):
        card = layout.card_box(index)
        image[card.y1:card.y2, card.x1:card.x2] = (225, 225, 225)
        cv2.line(image, (card.x1 + 10, card.y1 + 20), (card.x2 - 10, card.y1 + 20), (40, 40, 40), 4)
        box = layout.avatar_box(index, 0)
        rng = np.random.default_rng(index)
        image[box.y1:box.y2, box.x1:box.x2] = rng.integers(0, 256, (box.height, box.width, 3), dtype=np.uint8)
        image[box.y1:box.y1 + 5, box.x1:box.x2] = (180, 40, 240)
    located = layout.locate(image)
    assert [row.card_index for row in located] == list(range(count))


def test_gray_and_pink_frames_are_distinguished():
    layout = FixedLessonLayout()
    gray = np.full((58, 62, 3), 120, dtype=np.uint8)
    pink = gray.copy()
    pink[:5] = (180, 40, 240)
    pink[-5:] = (180, 40, 240)
    pink[:, :5] = (180, 40, 240)
    pink[:, -5:] = (180, 40, 240)
    assert not layout.affection_eligible(gray)
    assert layout.affection_eligible(pink)

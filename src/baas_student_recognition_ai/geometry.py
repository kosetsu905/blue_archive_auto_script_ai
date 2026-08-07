from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


@dataclass(frozen=True)
class AvatarSlot:
    card_index: int
    slot_index: int
    box: Box
    eligible: bool


class FixedLessonLayout:
    """Deterministic geometry for the new 1280x720 all-lessons dialog."""

    CARD_X = (129, 473, 817)
    CARD_Y = (181, 332, 483)
    CARD_SIZE = (337, 143)
    AVATAR_OFFSET = (20, 76)
    AVATAR_STEP_X = 72
    AVATAR_SIZE = (62, 58)

    def normalize(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.ndim != 3 or image.size == 0:
            raise ValueError("Invalid lesson screenshot")
        if image.shape[:2] == (720, 1280):
            return image
        return cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)

    def card_box(self, index: int) -> Box:
        if not 0 <= index < 9:
            raise IndexError(index)
        row, column = divmod(index, 3)
        x1, y1 = self.CARD_X[column], self.CARD_Y[row]
        width, height = self.CARD_SIZE
        return Box(x1, y1, x1 + width, y1 + height)

    def avatar_box(self, card_index: int, slot_index: int) -> Box:
        if not 0 <= slot_index < 3:
            raise IndexError(slot_index)
        card = self.card_box(card_index)
        width, height = self.AVATAR_SIZE
        x1 = card.x1 + self.AVATAR_OFFSET[0] + slot_index * self.AVATAR_STEP_X
        y1 = card.y1 + self.AVATAR_OFFSET[1]
        return Box(x1, y1, x1 + width, y1 + height)

    @staticmethod
    def card_present(patch: np.ndarray) -> bool:
        if patch.shape[0] < 100 or patch.shape[1] < 200:
            return False
        upper = patch[5:65, 10:-10]
        return float(upper.mean()) > 180.0 and float(upper.std()) > 22.0

    @staticmethod
    def avatar_present(patch: np.ndarray) -> bool:
        if patch.shape[0] < 20 or patch.shape[1] < 20:
            return False
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        return float(patch.std()) > 25.0 and float(hsv[:, :, 1].mean()) > 15.0

    @staticmethod
    def affection_eligible(patch: np.ndarray) -> bool:
        """Classify the frame, not the portrait artwork, as pink or gray."""
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        height, width = patch.shape[:2]
        frame = np.zeros((height, width), dtype=np.uint8)
        frame[:5, :] = 1
        frame[-5:, :] = 1
        frame[:, :5] = 1
        frame[:, -5:] = 1
        pixels = hsv[frame.astype(bool)]
        if len(pixels) == 0:
            return False
        pink = (
            (pixels[:, 0] >= 145)
            & (pixels[:, 0] <= 179)
            & (pixels[:, 1] >= 65)
            & (pixels[:, 2] >= 140)
        )
        colorful = pixels[:, 1] >= 55
        return float(pink.mean()) >= 0.025 or float(colorful.mean()) >= 0.18

    def locate(self, image: np.ndarray) -> list[AvatarSlot]:
        normalized = self.normalize(image)
        result: list[AvatarSlot] = []
        for card_index in range(9):
            card = self.card_box(card_index)
            card_patch = normalized[card.y1:card.y2, card.x1:card.x2]
            if not self.card_present(card_patch):
                continue
            for slot_index in range(3):
                box = self.avatar_box(card_index, slot_index)
                patch = normalized[box.y1:box.y2, box.x1:box.x2]
                if self.avatar_present(patch):
                    result.append(
                        AvatarSlot(
                            card_index=card_index,
                            slot_index=slot_index,
                            box=box,
                            eligible=self.affection_eligible(patch),
                        )
                    )
        return result


from enum import Enum


class SegmentationType(str, Enum):
    SEMANTIC = "semantic"
    INSTANCE = "instance"
    PANOPTIC = "panoptic"

    @property
    def pretty(self) -> str:
        return self.value.capitalize()

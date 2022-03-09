from enum import Enum


class ModelName(Enum):
    BASELINE = 1
    IMPROVED = 2


class LossFeature(Enum):
    TITLE = 0
    INGREDIENT_EMBEDDING = 1
    INGREDIENT = 2
    MASK = 3
    INSTRUCTIONS = 4

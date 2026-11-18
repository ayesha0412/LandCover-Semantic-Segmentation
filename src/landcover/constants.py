"""Dataset-level constants for LandCover.ai.

Class indices follow the LandCover.ai mask encoding. They are verified
empirically by ``scripts/inspect_dataset.py`` rather than assumed.
"""

CLASS_NAMES = ("background", "building", "woodland", "water", "road")
NUM_CLASSES = len(CLASS_NAMES)
IGNORE_INDEX = 255
TILE_SIZE = 512
SPLITS = ("train", "val", "test")

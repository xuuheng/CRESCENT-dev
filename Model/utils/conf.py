from pathlib import Path


depth=40
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = {
    "DATA_BASE_DIR": str(PROJECT_ROOT / "preprocess" / "Version0209" / "output" / "bin_with_case"),
    "FILE_EXTENSION": ".txt",
    "FEATURE_COL_START": 3,
    "FIXED_FEATURE_COLS": depth,
    "FEATURE_COL_END": 3 + depth,
    "SCALE_INPUT_SHAPES": [(100, depth), (500, depth), (2000, depth)],
    "TYPE_CENTERS": {
        # 该部分在本函数中不再使用
    },
    "OUTPUT_DIR": str(PROJECT_ROOT / "GeneratedSamples")
}

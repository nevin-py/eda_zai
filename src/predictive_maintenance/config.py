ENGINE_ID_COL = "engine_id"
CYCLE_COL = "cycle"
SETTING_COLS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
RUL_COL = "RUL"
FAILURE_SOON_COL = "failure_soon"

FEATURE_WINDOWS = (5, 10)
RUL_CLIP_VALUE = 125
FAILURE_THRESHOLD = 30

# JUPYTER CELL — feature: fvg_creation_flag_1
FEATURE_CODE = "fvg_creation_flag_1"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Fair Value Gap Creation Flag (1 bar)
    Description:
      Flags 1 if a fair value gap (FVG) is created, defined as a large price movement gap.
    Formula / method (brief):
      - A gap is formed when the open price is significantly different from the close price.
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    price_gap = abs(g["open"] - g["close"].shift(1))

    # Set threshold for large gap (FVG), e.g., 0.01 or any custom logic
    threshold = 0.01

    fvg_flag = (price_gap > threshold).astype(int)

    s = pd.Series(fvg_flag, index=g.index, name=FEATURE_CODE)
    return s
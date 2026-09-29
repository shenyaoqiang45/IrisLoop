#!/usr/bin/env python
"""Camera-projector homography calibration (flat wall). See irisloop/calibrate.py.

    python tools/run_calibration.py                      # full pipeline
    python tools/run_calibration.py solve --points ...   # offline re-solve
    python tools/run_calibration.py map --x 640 --y 360  # cam -> proj
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from irisloop.calibrate import main

if __name__ == "__main__":
    sys.exit(main())

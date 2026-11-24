"""Verify Python COMPhysics ('interp') open-loop parity against MATLAB validate_com_fig2."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nn_com.evaluation.parity_check import check_matlab_python_parity


if __name__ == "__main__":
    check_matlab_python_parity(REPO_ROOT)

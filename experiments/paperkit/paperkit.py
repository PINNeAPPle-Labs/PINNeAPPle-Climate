"""Shared figure style + PDF paper builder, now provided by the PINNeAPPle library.

The implementation lives in ``pinneapple_tools.reporting.paperkit`` (PINNeAPPle repo). This module keeps
``from paperkit import ...`` working for the paper scripts here: it imports the installed library, or the
sibling ``PINNeAPPle`` checkout the experiment scripts already use.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # the paper scripts run headless

try:
    import pinneapple_tools.reporting.paperkit  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "PINNeAPPle"))

from pinneapple_tools.reporting.paperkit import *  # noqa: E402,F401,F403
from pinneapple_tools.reporting.paperkit import __all__  # noqa: E402,F401
from pinneapple_tools.reporting.paperkit import para as _para  # noqa: E402,F401

"""Allow pytest to run from the repository root as documented for all MPs."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

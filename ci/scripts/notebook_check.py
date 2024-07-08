"""Scripts for Jupyter Notebooks.

* Checks whether the given directory has Notebooks with executed code cells not or not.
"""

import json
import sys
from pathlib import Path


def _notebook_has_outputs(path: Path) -> bool:
    with open(path, encoding="utf-8") as f:
        notebook = json.load(f)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code" and cell.get("execution_count") is not None:
            return True
    return False


if __name__ == "__main__":
    root = sys.argv[1]
    paths = Path(root).glob("**/*.ipynb")
    has_outputs = [str(p.resolve()) for p in paths if _notebook_has_outputs(p)]
    if has_outputs:
        paths = "\n".join(has_outputs)
        sys.exit(f"Found executed code cells in the following files:\n{paths}")

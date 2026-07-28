# /// script
# dependencies = [
#   "numpy", "pandas", "matplotlib", "scikit-learn", "torch",
#   "lightgbm", "catboost", "astroML", "pytabkit", "tabpfn",
#   "probmetrics", "autogluon", "xrfm", "tabicl", "huggingface-hub"
# ]
# ///
"""Clean CPU smoke test for the camera-ready classifier release.

The release is cloned at its published revision.  The only runtime overlay is
the same repository's ``experiments_general/code/ERT.py``, required because the
classifier driver imports it but the file is absent from that directory.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

from huggingface_hub import HfApi


PIN = "39a99dcad92205a15d93f2c5fec40c76540abf1c"
ROOT = Path("/tmp/conditional-coverage-source")
OUT = Path("/tmp/vaap-source-smoke.json")


def main() -> None:
    result = {"pin": PIN, "overlay": "experiments_general/code/ERT.py", "status": "failed"}
    try:
        subprocess.run(
            ["git", "clone", "https://github.com/ElSacho/Conditional_Coverage_Estimation.git", str(ROOT)],
            check=True,
        )
        subprocess.run(["git", "-C", str(ROOT), "checkout", "--detach", PIN], check=True)
        code = ROOT / "experiments/experiments_classifier_benchmark/code"
        source_ert = ROOT / "experiments/experiments_general/code/ERT.py"
        shutil.copy2(source_ert, code / "ERT.py")
        sys.path.insert(0, str(code))
        spec = importlib.util.spec_from_file_location("released_driver", code / "_generate_simultaneous_experiments.py")
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        result["status"] = "import-ok"
        result["torch_cuda_available"] = bool(module.torch.cuda.is_available())
    except Exception as error:
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        result["traceback"] = traceback.format_exc()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    HfApi(token=os.environ["HF_TOKEN"]).upload_file(
        path_or_fileobj=str(OUT),
        path_in_repo="vaApZm6MKM/source-smoke.json",
        repo_id="DineshAI/jobs-artifacts",
        repo_type="dataset",
    )
    print(json.dumps(result, indent=2))
    if result["status"] != "import-ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

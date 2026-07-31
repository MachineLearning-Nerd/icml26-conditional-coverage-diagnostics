"""Entry point for the one fixed run command, `bash repro/run.sh`.

The command takes no arguments.  What runs is `repro/config/stage.json` on the
current branch, so every node of the experiment tree executes the same command
over different committed configuration.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import traceback
import warnings

from .emit import artifact, note
from .provenance import Timer, snapshot

CONFIG_PATH = Path("repro/config/stage.json")


def _stages():
    from . import (stage_algorithm1, stage_convergence, stage_decomposition,
                   stage_principle, stage_smoke)
    registry = {
        "smoke": stage_smoke.run,
        "claim1_principle": stage_principle.run,
        "claim3_convergence": stage_convergence.run,
        "claim4_decomposition": stage_decomposition.run,
        "claim6_algorithm1": stage_algorithm1.run,
    }
    try:  # optional stages carry heavier imports
        from . import stage_table2
        registry["claim2_table2"] = stage_table2.run
    except ImportError as error:  # pragma: no cover - surfaced in the log
        note(f"stage claim2_table2 unavailable: {error}")
    try:
        from . import stage_table4
        registry["claim5_table4"] = stage_table4.run
    except ImportError as error:  # pragma: no cover
        note(f"stage claim5_table4 unavailable: {error}")
    return registry


def main() -> int:
    warnings.filterwarnings("ignore")
    config = json.loads(CONFIG_PATH.read_text())

    import torch
    torch.set_num_threads(int(config.get("torch_threads", 8)))
    stages = config.get("stages") or [config["stage"]]
    provenance = snapshot()

    note(f"stage.json -> {json.dumps(config, sort_keys=True)}")
    note(f"git {provenance['git_sha'][:12]} on {provenance['git_branch']}, "
         f"{provenance['cpu_allocation']} usable CPUs, python {provenance['python']}")
    artifact("run/provenance.json", provenance | {"config": config})

    registry = _stages()
    failures = []
    for name in stages:
        if name not in registry:
            note(f"FATAL: unknown stage {name!r}; known stages {sorted(registry)}")
            return 2
        note(f"===== stage {name} starting =====")
        with Timer() as timer:
            try:
                registry[name](config.get("params", {}).get(name, {}))
            except Exception:
                traceback.print_exc()
                failures.append(name)
        note(f"===== stage {name} finished in {timer.wall_s:.1f}s "
             f"(cpu {timer.cpu_s:.1f}s) =====")

    if failures:
        note(f"FAILED stages: {failures}")
        return 1
    note("all stages completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

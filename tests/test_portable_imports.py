import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_portable_runtime_imports_without_sibling_vst_rl():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from recurrent.skill_opd import validate_opd_teacher_cache; "
            "from verl.trainer.ppo.skill_opd_loss import compute_opd_advantage_weights",
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )

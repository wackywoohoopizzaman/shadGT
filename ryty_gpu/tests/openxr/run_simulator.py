# SPDX-License-Identifier: GPL-2.0-or-later
"""Run the stereo fixture with an isolated OpenXR Simulator profile for each mirror mode."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from verify import verify


# FEAT-SGT-051: give each mirror mode its own runtime profile and capture requests.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("runtime_manifest", type=Path)
    parser.add_argument("--output", type=Path, help="parent directory for a fresh capture run")
    args = parser.parse_args()
    executable = args.executable.resolve(strict=True)
    manifest = args.runtime_manifest.resolve(strict=True)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="gt7-stereo-", dir=args.output)).resolve()
    print(f"Simulator captures in {root}", flush=True)
    for mode in ("left", "right", "both", "off"):
        profile = root / mode
        captures = profile / "OpenXR-Simulator"
        captures.mkdir(parents=True)
        shutil.copyfile(Path(__file__).with_name("settings.json"), captures / "settings.json")
        env = dict(os.environ, XR_RUNTIME_JSON=str(manifest), LOCALAPPDATA=str(profile),
                   SHARED_OPENXR="1", SHADGT_OPENXR_MIRROR_EYE=mode)
        print(f"Testing {mode} mirror", flush=True)
        subprocess.run([str(executable), "--simulator"], env=env, check=True, timeout=120)
        verify(captures)


if __name__ == "__main__":
    main()

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    crate = ROOT / "native"
    env = dict(os.environ, PYO3_PYTHON=sys.executable)
    subprocess.run(["cargo", "build", "--release"], cwd=crate, env=env, check=True)
    lib = crate / "target" / "release" / ("_rules.dll" if os.name == "nt" else "lib_rules.so")
    if sys.platform == "darwin":
        lib = lib.with_suffix(".dylib")
    out = ROOT / "bsgen" / ("_rules.pyd" if os.name == "nt" else "_rules.abi3.so")
    tmp = out.with_name(out.name + ".new")
    shutil.copyfile(lib, tmp)
    os.replace(tmp, out)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
"""Execute actual ABCDF and joint kernel helpers using a 32-thread CPU SIMT shim.

Requires g++ with C++20. Does not import torch or model NPU cache/address spaces.
Run directly; the parent pytest suite needs torch and is not available on every PC.
"""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parents[3]
    kernel_dir = root / "csrc/attention/sparse_kv_plan/op_kernel"
    kernel = (kernel_dir / "sparse_kv_plan_apt.cpp").read_text()
    helpers = kernel[kernel.index("namespace {") : kernel.index("/**\n * SIMT 调度层")]
    helpers = helpers.replace('#include "sparse_kv_joint_plan.h"', (kernel_dir / "sparse_kv_joint_plan.h").read_text())
    code = (here / "cpu_driver.cpp").read_text().replace("// INSERT_KERNEL", helpers)
    with tempfile.TemporaryDirectory(prefix="joint-plan-") as temp:
        source = Path(temp) / "test.cpp"
        binary = Path(temp) / "test"
        source.write_text(code)
        cmd = ["g++", "-std=c++20", "-O1", "-g", "-pthread"]
        if args.sanitize:
            cmd += ["-fsanitize=address,undefined"]
        subprocess.run([*cmd, str(source), "-o", str(binary)], check=True)
        # LeakSanitizer cannot run under the desktop sandbox's ptrace.
        env = dict(os.environ)
        if args.sanitize:
            env["ASAN_OPTIONS"] = "detect_leaks=0"
        subprocess.run([str(binary)], check=True, env=env)


if __name__ == "__main__":
    main()

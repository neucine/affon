#!/usr/bin/env python3
"""Build both Affon CPU runtimes, measure Llama, and verify routing and parity."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

TOLERANCE = {"absolute": 3e-4, "relative": 3e-4}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def machine_metadata() -> dict:
    metadata = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "python": platform.python_version(),
    }
    if platform.system() == "Darwin":
        for key, command in (
            ("cpu", ["sysctl", "-n", "machdep.cpu.brand_string"]),
            ("system_memory_bytes", ["sysctl", "-n", "hw.memsize"]),
            ("os_build", ["sw_vers", "-buildVersion"]),
        ):
            try:
                value = subprocess.check_output(command, text=True).strip()
                metadata[key] = int(value) if key == "system_memory_bytes" else value
            except (OSError, subprocess.CalledProcessError, ValueError):
                metadata[key] = None
    return metadata


def run_logged(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> None:
    print("+ " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
    log_path.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); inspect {log_path}")


def compare_values(left: list[float], right: list[float]) -> dict:
    if len(left) != len(right):
        raise ValueError(f"output lengths differ: {len(left)} != {len(right)}")
    maximum_difference = 0.0
    maximum_relative = 0.0
    failures = 0
    for index, (actual, expected) in enumerate(zip(left, right)):
        difference = abs(actual - expected)
        limit = TOLERANCE["absolute"] + TOLERANCE["relative"] * abs(expected)
        maximum_difference = max(maximum_difference, difference)
        maximum_relative = max(maximum_relative, difference / max(abs(expected), 1e-30))
        if not math.isfinite(actual) or not math.isfinite(expected) or difference > limit:
            failures += 1
            if failures <= 5:
                print(
                    f"parity failure at {index}: actual={actual}, expected={expected}, "
                    f"difference={difference}, limit={limit}",
                    file=sys.stderr,
                )
    return {
        "elements": len(left),
        "failures": failures,
        "maximum_absolute_difference": maximum_difference,
        "maximum_relative_difference": maximum_relative,
        "tolerance": TOLERANCE,
    }


def parser() -> argparse.ArgumentParser:
    default_output = Path("/tmp") / f"affon-cyna-performance-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    argument = argparse.ArgumentParser(description=__doc__)
    argument.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    argument.add_argument("--output", type=Path, default=default_output)
    argument.add_argument(
        "--cyna-lib-dir",
        type=Path,
        default=Path(os.environ.get("AFFON_CYNA_LIB_DIR", Path.home() / "Private/cyna/zig-out/lib")),
    )
    argument.add_argument(
        "--compute-core-lib-dir",
        type=Path,
        default=Path(os.environ.get("AFFON_COMPUTE_CORE_LIB_DIR", Path.home() / "Private/compute-core/build")),
    )
    argument.add_argument("--warmups", type=int, default=2)
    argument.add_argument("--samples", type=int, default=7)
    argument.add_argument("--executions-per-sample", type=int, default=1)
    argument.add_argument("--skip-build", action="store_true")
    argument.add_argument("--cyna-affon", type=Path)
    argument.add_argument("--zig-affon", type=Path)
    return argument


def main() -> int:
    arguments = parser().parse_args()
    if not 0 <= arguments.warmups <= 100:
        parser().error("--warmups must be from 0 to 100")
    if not 3 <= arguments.samples <= 100:
        parser().error("--samples must be from 3 to 100")
    if not 1 <= arguments.executions_per_sample <= 100:
        parser().error("--executions-per-sample must be from 1 to 100")

    repo = arguments.repo.resolve()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    build_dir = output / "build"
    run_dir = output / "runs"
    build_dir.mkdir(exist_ok=True)
    run_dir.mkdir(exist_ok=True)
    zig = os.environ.get("AFFON_ZIG") or shutil.which("zig")
    if not zig:
        raise RuntimeError("zig was not found; set AFFON_ZIG")
    environment = {**os.environ}
    if arguments.skip_build:
        if not arguments.cyna_affon or not arguments.zig_affon:
            raise RuntimeError("--skip-build requires --cyna-affon and --zig-affon")
        executables = {
            "cyna": arguments.cyna_affon.resolve(),
            "zig": arguments.zig_affon.resolve(),
        }
    else:
        if not arguments.cyna_lib_dir.is_dir():
            raise RuntimeError(f"Cyna library directory not found: {arguments.cyna_lib_dir}")
        if not arguments.compute_core_lib_dir.is_dir():
            raise RuntimeError(f"compute-core library directory not found: {arguments.compute_core_lib_dir}")
        prefixes = {"cyna": build_dir / "cyna", "zig": build_dir / "zig"}
        run_logged(
            [
                zig,
                "build",
                "-Dcompute-cyna",
                f"-Dcyna-lib-dir={arguments.cyna_lib_dir.resolve()}",
                f"-Dcompute-core-lib-dir={arguments.compute_core_lib_dir.resolve()}",
                "-Doptimize=ReleaseFast",
                "--prefix",
                str(prefixes["cyna"]),
            ],
            cwd=repo,
            env=environment,
            log_path=build_dir / "cyna-build.log",
        )
        run_logged(
            [zig, "build", "-Doptimize=ReleaseFast", "--prefix", str(prefixes["zig"])],
            cwd=repo,
            env=environment,
            log_path=build_dir / "zig-build.log",
        )
        executables = {name: prefix / "bin/affon" for name, prefix in prefixes.items()}

    for executable in executables.values():
        if not executable.is_file():
            raise RuntimeError(f"Affon executable not found: {executable}")

    expectations = {
        "cyna": {"backend": "cyna", "reason": "lowered"},
        "zig": {"backend": "zig", "reason": "cyna_disabled"},
    }
    reports = {}
    for backend in ("cyna", "zig"):
        report_path = run_dir / f"{backend}.json"
        log_path = run_dir / f"{backend}.log"
        run_env = {
            **environment,
            "AFFON_CYNA_BENCH_REPORT": str(report_path),
            "AFFON_CYNA_BENCH_EXPECTED_BACKEND": expectations[backend]["backend"],
            "AFFON_CYNA_BENCH_EXPECTED_REASON": expectations[backend]["reason"],
            "AFFON_CYNA_BENCH_BUILD_PROFILE": f"ReleaseFast/{backend}",
            "AFFON_CYNA_BENCH_WARMUPS": str(arguments.warmups),
            "AFFON_CYNA_BENCH_SAMPLES": str(arguments.samples),
            "AFFON_CYNA_BENCH_EXECUTIONS_PER_SAMPLE": str(arguments.executions_per_sample),
        }
        run_logged(
            [str(executables[backend]), "test/benchmark/cyna-routing/performance.ts"],
            cwd=repo,
            env=run_env,
            log_path=log_path,
        )
        reports[backend] = json.loads(report_path.read_text())
        report = reports[backend]
        for workload in ("prefill", "decode"):
            route = report["routes"][workload]
            if route["backend"] != expectations[backend]["backend"]:
                raise RuntimeError(f"{backend} {workload} backend route mismatch")
            if route["reason"] != expectations[backend]["reason"]:
                raise RuntimeError(f"{backend} {workload} routing reason mismatch")

    parity = {
        "prefill_logits": compare_values(
            reports["cyna"]["reference"]["prefill_logits"],
            reports["zig"]["reference"]["prefill_logits"],
        ),
        "decode_logits": compare_values(
            reports["cyna"]["reference"]["decode_logits"],
            reports["zig"]["reference"]["decode_logits"],
        ),
    }
    if any(result["failures"] for result in parity.values()):
        raise RuntimeError("Cyna and Zig outputs exceeded the f32 tolerance")

    ratios = {}
    for workload in ("prefill", "decode"):
        cyna_ms = reports["cyna"]["measurement"][workload]["median_ms"]
        zig_ms = reports["zig"]["measurement"][workload]["median_ms"]
        ratios[workload] = {
            "cyna_over_zig": cyna_ms / zig_ms,
            "zig_over_cyna": zig_ms / cyna_ms,
        }
    summary = {
        "schema_version": 1,
        "format": "affon-cyna-llama-performance-summary/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "machine": machine_metadata(),
        "revisions": {
            "affon": git_revision(repo),
            "cyna": git_revision(arguments.cyna_lib_dir.resolve()) if not arguments.skip_build else None,
            "compute_core": git_revision(arguments.compute_core_lib_dir.resolve()) if not arguments.skip_build else None,
        },
        "executables": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in executables.items()
        },
        "source": {
            "benchmark_sha256": sha256(Path(__file__).with_name("performance.ts")),
            "llama_model_sha256": sha256(repo / "packages/@affon/models/src/llama/model.ts"),
        },
        "measurement": {
            "warmups": arguments.warmups,
            "samples": arguments.samples,
            "executions_per_sample": arguments.executions_per_sample,
            "order": ["cyna", "zig"],
            "processes": "Each runtime ran in a fresh sequential process.",
        },
        "results": {
            backend: {
                workload: reports[backend]["measurement"][workload]
                for workload in ("prefill", "decode")
            }
            for backend in ("cyna", "zig")
        },
        "ratios": ratios,
        "parity": parity,
        "limitations": [
            "The synthetic weights measure runtime speed and numerical agreement, not trained-model quality.",
            "This one CPU workload does not establish broad model, shape, dtype, or backend coverage.",
            "The runs are sequential; uncontrolled host activity can still affect wall-clock timing.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"summary: {output / 'summary.json'}")
    for workload in ("prefill", "decode"):
        cyna_ms = summary["results"]["cyna"][workload]["median_ms"]
        zig_ms = summary["results"]["zig"][workload]["median_ms"]
        print(f"{workload}: Cyna {cyna_ms:.3f} ms, Zig {zig_ms:.3f} ms, ratio {cyna_ms / zig_ms:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

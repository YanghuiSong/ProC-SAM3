from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
EVAL_SCRIPT = REPO_ROOT / "eval.py"
DEFAULT_LOG_ROOT = REPO_ROOT / "batch_eval_runs"
METRIC_KEYS = ("aAcc", "mIoU", "mAcc", "AP50", "AP75")
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    config_path: Path


DATASET_SPECS = OrderedDict(
    (
        ("isaid", DatasetSpec("isaid", "iSAID", REPO_ROOT / "configs" / "cfg_iSAID_controlled.py")),
        ("loveda", DatasetSpec("loveda", "LoveDA", REPO_ROOT / "configs" / "cfg_loveda_controlled.py")),
        (
            "openearthmap",
            DatasetSpec(
                "openearthmap",
                "OpenEarthMap",
                REPO_ROOT / "configs" / "cfg_openearthmap_controlled.py",
            ),
        ),
        ("potsdam", DatasetSpec("potsdam", "Potsdam", REPO_ROOT / "configs" / "cfg_potsdam_controlled.py")),
        ("uavid", DatasetSpec("uavid", "UAVid", REPO_ROOT / "configs" / "cfg_uavid_controlled.py")),
        ("udd5", DatasetSpec("udd5", "UDD5", REPO_ROOT / "configs" / "cfg_udd5_controlled.py")),
        (
            "vaihingen",
            DatasetSpec("vaihingen", "Vaihingen", REPO_ROOT / "configs" / "cfg_vaihingen_controlled.py"),
        ),
        ("vdd", DatasetSpec("vdd", "VDD", REPO_ROOT / "configs" / "cfg_vdd_controlled.py")),
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Safely run the eight controlled evaluation configs one at a time, "
            "with optional dataset selection and aggregate reporting."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["all"],
        help=(
            "Datasets to run. Accepts space-separated names or comma-separated "
            "lists. Use 'all' for the full suite."
        ),
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Print the supported dataset keys and exit.",
    )
    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python executable used to invoke eval.py.",
    )
    parser.add_argument(
        "--percent",
        type=float,
        default=100.0,
        help="Forwarded to eval.py --percent.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Forwarded to eval.py --seed.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="Forwarded to eval.py --launcher.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="*",
        default=[],
        help="Extra cfg-options forwarded verbatim to eval.py.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=5.0,
        help="Idle time inserted between datasets to avoid back-to-back GPU load spikes.",
    )
    parser.add_argument(
        "--timeout-hours",
        type=float,
        default=0.0,
        help="Per-dataset timeout in hours. Use 0 to disable timeouts.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=4,
        help=(
            "Cap common BLAS/OpenMP thread pools for the child process. "
            "Use 0 to keep the current environment unchanged."
        ),
    )
    parser.add_argument(
        "--log-root",
        default=str(DEFAULT_LOG_ROOT),
        help="Directory used for run logs and aggregate summaries.",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Optional name for this batch run. Defaults to a timestamp.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop after the first failed dataset instead of continuing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved commands without executing them.",
    )
    parser.add_argument(
        "--out-root",
        default="",
        help=(
            "Optional root directory forwarded to eval.py --out, with one child "
            "directory created per dataset."
        ),
    )
    return parser.parse_args()


def normalize_dataset_selection(raw_items: list[str]) -> list[str]:
    tokens: list[str] = []
    for item in raw_items:
        for part in item.split(","):
            token = part.strip().lower()
            if token:
                tokens.append(token)

    if not tokens or "all" in tokens:
        return list(DATASET_SPECS.keys())

    unknown = [token for token in tokens if token not in DATASET_SPECS]
    if unknown:
        raise ValueError(
            "Unknown dataset key(s): "
            + ", ".join(sorted(set(unknown)))
            + ". Use --list-datasets to inspect valid names."
        )

    ordered: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token not in seen:
            seen.add(token)
            ordered.append(token)
    return ordered


def build_child_env(cpu_threads: int) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if cpu_threads > 0:
        for name in THREAD_ENV_VARS:
            env[name] = str(cpu_threads)
    return env


def get_results_txt_path(config_path: Path) -> Path:
    config_stem = config_path.stem
    return REPO_ROOT / "work_dirs" / config_stem / "results.txt"


def get_file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def read_appended_text(path: Path, start_offset: int) -> str:
    if not path.exists():
        return ""

    current_size = path.stat().st_size
    effective_offset = 0 if start_offset < 0 or start_offset > current_size else start_offset
    with path.open("rb") as handle:
        handle.seek(effective_offset)
        return handle.read().decode("utf-8", errors="replace")


def coerce_value(text: str) -> Any:
    value = text.strip()
    if not value:
        return value
    try:
        return float(value)
    except ValueError:
        return value


def parse_metrics_from_results_text(text: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        metrics[key.strip()] = coerce_value(value)
    return metrics


def parse_metrics_from_console(lines: list[str]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    metric_pattern = re.compile(r"\b(aAcc|mIoU|mAcc|AP50|AP75):\s*([0-9]+(?:\.[0-9]+)?)")
    for line in lines:
        for key, value in metric_pattern.findall(line):
            metrics[key] = float(value)
    return metrics


def render_command(command: list[str]) -> str:
    return subprocess.list2cmdline([str(part) for part in command])


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def terminate_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    proc.kill()
    proc.wait(timeout=10)


def stream_process(
    command: list[str],
    log_path: Path,
    env: dict[str, str],
    timeout_seconds: float,
) -> tuple[int, list[str], bool]:
    sentinel = object()
    queue: Queue[Any] = Queue()
    recent_lines: deque[str] = deque(maxlen=400)
    timed_out = False

    proc = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    def reader_worker() -> None:
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                queue.put(line)
        finally:
            proc.stdout.close()
            queue.put(sentinel)

    reader_thread = Thread(target=reader_worker, daemon=True)
    reader_thread.start()

    deadline = time.monotonic() + timeout_seconds if timeout_seconds > 0 else None

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"COMMAND: {render_command(command)}\n")
        log_file.write(f"STARTED_AT: {datetime.now().isoformat(timespec='seconds')}\n\n")

        try:
            saw_sentinel = False
            while True:
                if deadline is not None and time.monotonic() > deadline:
                    timed_out = True
                    terminate_process(proc)
                    break

                try:
                    item = queue.get(timeout=0.5)
                except Empty:
                    if proc.poll() is not None and not reader_thread.is_alive():
                        break
                    continue

                if item is sentinel:
                    saw_sentinel = True
                    if proc.poll() is not None:
                        break
                    continue

                print(item, end="")
                log_file.write(item)
                log_file.flush()
                recent_lines.append(item)

                if saw_sentinel and proc.poll() is not None:
                    break
        except KeyboardInterrupt:
            terminate_process(proc)
            raise

        return_code = proc.wait()
        log_file.write(
            f"\nRETURN_CODE: {return_code}\nTIMED_OUT: {str(timed_out).lower()}\n"
        )

    reader_thread.join(timeout=2)
    return return_code, list(recent_lines), timed_out


def compute_average_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    averages: dict[str, float] = {}
    for key in METRIC_KEYS:
        values = [
            float(record[key])
            for record in records
            if record.get("status") == "ok" and isinstance(record.get(key), (int, float))
        ]
        if values:
            averages[key] = sum(values) / len(values)
    return averages


def count_status(records: list[dict[str, Any]], *statuses: str) -> int:
    expected = set(statuses)
    return sum(1 for record in records if record.get("status") in expected)


def write_summary_files(run_dir: Path, selected_keys: list[str], records: list[dict[str, Any]]) -> None:
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "selected_datasets": selected_keys,
        "completed": count_status(records, "ok"),
        "failed": count_status(records, "failed", "timeout"),
        "dry_run": count_status(records, "dry_run"),
        "average_metrics": compute_average_metrics(records),
        "records": records,
    }

    summary_json = run_dir / "summary.json"
    summary_csv = run_dir / "summary.csv"
    summary_md = run_dir / "summary.md"

    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)

    fieldnames = [
        "dataset_key",
        "dataset_name",
        "status",
        "duration_seconds",
        "config",
        "log_path",
        "results_txt_path",
        *METRIC_KEYS,
        "error",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})

    lines = [
        "# Controlled Evaluation Summary",
        "",
        f"- Generated at: {summary['generated_at']}",
        f"- Selected datasets: {', '.join(selected_keys)}",
        f"- Completed: {summary['completed']}",
        f"- Failed: {summary['failed']}",
        f"- Dry run: {summary['dry_run']}",
        "",
        "## Average Metrics",
        "",
    ]
    if summary["average_metrics"]:
        for key, value in summary["average_metrics"].items():
            lines.append(f"- {key}: {value:.4f}")
    else:
        lines.append("- No successful runs yet.")

    lines.extend(
        [
            "",
            "## Per-Dataset Results",
            "",
            "| Dataset | Status | mIoU | AP50 | AP75 | Duration(s) |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for record in records:
        lines.append(
            "| {dataset_name} | {status} | {mIoU} | {AP50} | {AP75} | {duration_seconds:.2f} |".format(
                dataset_name=record["dataset_name"],
                status=record["status"],
                mIoU=format_metric(record.get("mIoU")),
                AP50=format_metric(record.get("AP50")),
                AP75=format_metric(record.get("AP75")),
                duration_seconds=float(record.get("duration_seconds", 0.0)),
            )
        )

    with summary_md.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def format_metric(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"
    return "-"


def run_single_dataset(
    spec: DatasetSpec,
    args: argparse.Namespace,
    run_dir: Path,
    child_env: dict[str, str],
) -> dict[str, Any]:
    command: list[str] = [
        str(Path(args.python_exe)),
        str(EVAL_SCRIPT),
        str(spec.config_path),
        "--launcher",
        args.launcher,
        "--percent",
        str(args.percent),
        "--seed",
        str(args.seed),
    ]

    if args.out_root:
        dataset_out_dir = Path(args.out_root).expanduser().resolve() / spec.key
        dataset_out_dir.mkdir(parents=True, exist_ok=True)
        command.extend(["--out", str(dataset_out_dir)])

    if args.cfg_options:
        command.append("--cfg-options")
        command.extend(args.cfg_options)

    log_path = run_dir / f"{spec.key}.log"
    results_txt_path = get_results_txt_path(spec.config_path)
    result: dict[str, Any] = {
        "dataset_key": spec.key,
        "dataset_name": spec.display_name,
        "config": display_path(spec.config_path),
        "log_path": display_path(log_path),
        "results_txt_path": display_path(results_txt_path),
        "status": "pending",
        "duration_seconds": 0.0,
        "error": "",
    }

    print(f"\n=== [{spec.display_name}] ===")
    print(f"CONFIG: {result['config']}")
    print(f"LOG: {result['log_path']}")
    print(f"COMMAND: {render_command(command)}")

    if args.dry_run:
        result["status"] = "dry_run"
        return result

    pre_run_size = get_file_size(results_txt_path)
    started = time.monotonic()
    return_code, recent_lines, timed_out = stream_process(
        command=command,
        log_path=log_path,
        env=child_env,
        timeout_seconds=max(args.timeout_hours, 0.0) * 3600.0,
    )
    result["duration_seconds"] = time.monotonic() - started

    appended_text = read_appended_text(results_txt_path, pre_run_size)
    parsed_metrics = parse_metrics_from_results_text(appended_text)
    if not parsed_metrics:
        parsed_metrics = parse_metrics_from_console(recent_lines)

    for key in METRIC_KEYS:
        if key in parsed_metrics:
            result[key] = parsed_metrics[key]
    for key in ("Model", "Dataset", "Data_Percent"):
        if key in parsed_metrics:
            result[key] = parsed_metrics[key]

    if return_code == 0 and not timed_out:
        result["status"] = "ok"
    elif timed_out:
        result["status"] = "timeout"
        result["error"] = f"Timed out after {args.timeout_hours:.2f} hour(s)."
    else:
        result["status"] = "failed"
        result["error"] = f"eval.py exited with code {return_code}."

    return result


def main() -> int:
    args = parse_args()

    if args.list_datasets:
        for key, spec in DATASET_SPECS.items():
            print(f"{key:<13} -> {display_path(spec.config_path)}")
        return 0

    try:
        selected_keys = normalize_dataset_selection(args.datasets)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    missing_configs = [
        spec.config_path for key, spec in DATASET_SPECS.items() if key in selected_keys and not spec.config_path.exists()
    ]
    if missing_configs:
        for path in missing_configs:
            print(f"ERROR: Missing config file: {path}", file=sys.stderr)
        return 2

    run_name = args.run_name.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.log_root).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    child_env = build_child_env(args.cpu_threads)
    records: list[dict[str, Any]] = []

    print("Batch evaluation safeguards:")
    print("- Runs only the eight whitelisted controlled configs.")
    print("- Executes one dataset at a time; no parallel jobs are spawned.")
    print("- Does not delete, move, or overwrite dataset files.")
    print("- Writes only logs and summaries under:", run_dir)
    print(
        "- eval.py still keeps its normal side effects under work_dirs/, results.xlsx, "
        "and prompt_pools/ if prompt pools must be generated."
    )

    for index, key in enumerate(selected_keys):
        spec = DATASET_SPECS[key]
        try:
            record = run_single_dataset(spec, args, run_dir, child_env)
        except KeyboardInterrupt:
            print("\nInterrupted. Writing partial summary before exit.")
            write_summary_files(run_dir, selected_keys, records)
            return 130

        records.append(record)
        write_summary_files(run_dir, selected_keys, records)

        status = record["status"]
        metric_text = ", ".join(
            f"{metric}={format_metric(record.get(metric))}" for metric in ("mIoU", "AP50", "AP75")
        )
        print(
            f"RESULT [{spec.display_name}] status={status}, "
            f"duration={record['duration_seconds']:.2f}s, {metric_text}"
        )

        if status != "ok" and args.stop_on_error:
            print("Stopping on first error because --stop-on-error was set.")
            break

        if index + 1 < len(selected_keys) and args.pause_seconds > 0 and not args.dry_run:
            print(f"Cooling down for {args.pause_seconds:.1f}s before the next dataset.")
            time.sleep(args.pause_seconds)

    averages = compute_average_metrics(records)
    print("\n=== Aggregate Summary ===")
    print(f"Run directory: {run_dir}")
    print(f"Completed: {count_status(records, 'ok')}")
    print(f"Failed: {count_status(records, 'failed', 'timeout')}")
    print(f"Dry run: {count_status(records, 'dry_run')}")
    if averages:
        for key in METRIC_KEYS:
            if key in averages:
                print(f"{key}: {averages[key]:.4f}")
    else:
        print("No successful runs to average.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

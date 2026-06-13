from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def value_after(
    arguments: list[str],
    flag: str,
    default: str,
) -> str:
    if flag not in arguments:
        return default
    index = arguments.index(flag)
    if index + 1 >= len(arguments):
        raise ValueError(f"Missing value for {flag}")
    return arguments[index + 1]


def remove_option(
    arguments: list[str],
    flag: str,
) -> tuple[list[str], str | None]:
    output: list[str] = []
    value: str | None = None
    index = 0
    while index < len(arguments):
        if arguments[index] == flag:
            if index + 1 >= len(arguments):
                raise ValueError(f"Missing value for {flag}")
            value = arguments[index + 1]
            index += 2
            continue
        output.append(arguments[index])
        index += 1
    return output, value


def infer_policy_mode(
    explicit: str | None,
    outputs_dir: Path,
) -> str:
    if explicit is not None:
        if explicit not in {"strict", "smoke"}:
            raise ValueError(
                "--semantics-mode must be strict or smoke"
            )
        return explicit

    normalized = str(outputs_dir).lower()
    return "smoke" if "smoke" in normalized else "strict"


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    raw_arguments = sys.argv[1:]
    pipeline_arguments, explicit_mode = remove_option(
        raw_arguments,
        "--semantics-mode",
    )

    outputs = Path(
        value_after(
            pipeline_arguments,
            "--outputs-dir",
            "outputs",
        )
    )
    reports = Path(
        value_after(
            pipeline_arguments,
            "--reports-dir",
            "reports",
        )
    )
    if not outputs.is_absolute():
        outputs = repo / outputs
    if not reports.is_absolute():
        reports = repo / reports

    policy_mode = infer_policy_mode(
        explicit_mode,
        outputs,
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "run_full_pipeline.py"),
            *pipeline_arguments,
        ],
        cwd=repo,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    completed = subprocess.run(
        [
            sys.executable,
            str(
                repo
                / "scripts"
                / "finalize_selection_launch_semantics.py"
            ),
            "--outputs-dir",
            str(outputs),
            "--reports-dir",
            str(reports),
            "--config",
            str(repo / "configs" / "model_selection.yaml"),
            "--policy-mode",
            policy_mode,
        ],
        cwd=repo,
        check=False,
    )
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()

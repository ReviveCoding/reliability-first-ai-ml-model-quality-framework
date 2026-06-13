from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

TEXT_SUFFIXES = {
    ".cff",
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".rst",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

REQUIRED_PATHS = (
    "README.md",
    "LICENSE",
    "SECURITY.md",
    "CONTRIBUTING.md",
    "CITATION.cff",
    ".gitignore",
    ".gitattributes",
    ".github/workflows/ci.yml",
    ".github/workflows/release-contract.yml",
    ".github/dependabot.yml",
    "pyproject.toml",
    "requirements-ci.txt",
    "scripts/verify_release_contract.py",
    "docs/ARCHITECTURE.md",
    "docs/CLAIM_BOUNDARIES.md",
    "docs/FINAL_MODEL_DECISION.md",
    "docs/EXPERIMENTAL_EVIDENCE.md",
    "docs/REPRODUCIBILITY.md",
    "docs/artifacts/final_evaluation/manifest.json",
    "docs/artifacts/final_evaluation/transformer_training_provenance.json",
    "docs/artifacts/final_evaluation/weighted_challenger_decision.json",
)

FORBIDDEN_PREFIXES = (
    ".cache/",
    ".mypy_cache/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".venv/",
    "backups/",
    "cw/",
    "mlruns/",
    "outputs/",
    "runs/",
    "sa/",
)

FORBIDDEN_PARTS = (
    ".egg-info/",
    "/__pycache__/",
)

FORBIDDEN_SUFFIXES = (
    ".ckpt",
    ".joblib",
    ".onnx",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
)

SECRET_PATTERNS = {
    "private_key": re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
    ),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "openai_style_key": re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
}

LOCAL_PATH_PATTERNS = {
    "windows_user_path": re.compile(
        r"(?i)\b[A-Z]:"
        r"(?:\\\\|\\)"
        r"Users"
        r"(?:\\\\|\\)"
        r"[^\\/\r\n]+"
        r"(?:\\\\|\\)"
    ),
    "mac_user_path": re.compile(
        r"(?<!https:)(?<!http:)/" r"Users/[^/\s]+/"
    ),
    "linux_home_path": re.compile(
        r"(?<!https:)(?<!http:)/" r"home/[^/\s]+/"
    ),
}


@dataclass
class Finding:
    level: str
    check: str
    message: str
    path: str | None = None


def run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def tracked_files(repo: Path) -> list[str]:
    result = run_git(repo, "ls-files", "-z")
    if result.returncode == 0:
        return sorted(path for path in result.stdout.split("\0") if path)

    excluded_roots = {".git", ".venv", "backups"}
    return sorted(
        path.relative_to(repo).as_posix()
        for path in repo.rglob("*")
        if path.is_file()
        and not any(part in excluded_roots for part in path.relative_to(repo).parts)
    )


def is_finite_json(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(is_finite_json(item) for item in value.values())
    if isinstance(value, list):
        return all(is_finite_json(item) for item in value)
    return True


def all_string_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from all_string_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from all_string_values(item)


def check_required_paths(repo: Path, findings: list[Finding]) -> None:
    for relative in REQUIRED_PATHS:
        if not (repo / relative).is_file():
            findings.append(
                Finding("ERROR", "required_path", "Required release file is missing.", relative)
            )


def check_tracked_hygiene(
    repo: Path, tracked: list[str], findings: list[Finding]
) -> None:
    for relative in tracked:
        normalized = relative.replace("\\", "/")
        lowered = normalized.lower()

        if any(lowered.startswith(prefix.lower()) for prefix in FORBIDDEN_PREFIXES):
            findings.append(
                Finding(
                    "ERROR",
                    "forbidden_tracked_path",
                    "Generated or local-only path is tracked.",
                    relative,
                )
            )

        padded = f"/{lowered}"
        if any(part.lower() in padded for part in FORBIDDEN_PARTS):
            findings.append(
                Finding(
                    "ERROR",
                    "forbidden_tracked_path",
                    "Generated Python metadata or cache is tracked.",
                    relative,
                )
            )

        if lowered.endswith(FORBIDDEN_SUFFIXES):
            findings.append(
                Finding(
                    "ERROR",
                    "forbidden_model_artifact",
                    "Model binary is tracked instead of documented as an external artifact.",
                    relative,
                )
            )

        path = repo / relative
        if path.is_file():
            size = path.stat().st_size
            if size >= 100 * 1024 * 1024:
                findings.append(
                    Finding(
                        "ERROR",
                        "file_size",
                        f"Tracked file is {size / 1024 / 1024:.2f} MiB.",
                        relative,
                    )
                )
            elif size >= 50 * 1024 * 1024:
                findings.append(
                    Finding(
                        "WARN",
                        "file_size",
                        f"Tracked file is {size / 1024 / 1024:.2f} MiB.",
                        relative,
                    )
                )


def check_text_hygiene(
    repo: Path, tracked: list[str], findings: list[Finding]
) -> None:
    for relative in tracked:
        path = repo / relative
        if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():
            continue

        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            findings.append(
                Finding(
                    "ERROR",
                    "utf8",
                    "Tracked text file is not valid UTF-8.",
                    relative,
                )
            )
            continue

        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append(
                    Finding(
                        "ERROR",
                        f"secret_{name}",
                        "Potential credential or private key pattern found.",
                        relative,
                    )
                )

        for name, pattern in LOCAL_PATH_PATTERNS.items():
            if pattern.search(text):
                findings.append(
                    Finding(
                        "ERROR",
                        f"local_path_{name}",
                        "Absolute user-specific local path found in a tracked text file.",
                        relative,
                    )
                )


def check_structured_files(
    repo: Path, tracked: list[str], findings: list[Finding]
) -> None:
    for relative in tracked:
        path = repo / relative
        suffix = path.suffix.lower()

        if suffix == ".json":
            try:
                value = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError) as exc:
                findings.append(
                    Finding("ERROR", "json_parse", f"Invalid JSON: {exc}", relative)
                )
                continue

            if not is_finite_json(value):
                findings.append(
                    Finding(
                        "ERROR",
                        "json_non_finite",
                        "JSON contains NaN or infinity.",
                        relative,
                    )
                )

        elif suffix == ".csv":
            try:
                with path.open("r", encoding="utf-8-sig", newline="") as handle:
                    rows = list(csv.reader(handle))
            except (OSError, csv.Error) as exc:
                findings.append(
                    Finding("ERROR", "csv_parse", f"Invalid CSV: {exc}", relative)
                )
                continue

            if not rows or not rows[0]:
                findings.append(
                    Finding("ERROR", "csv_empty", "CSV has no header row.", relative)
                )


def check_decision_contract(repo: Path, findings: list[Finding]) -> None:
    decision_path = repo / "docs/FINAL_MODEL_DECISION.md"
    if decision_path.is_file():
        decision_text = decision_path.read_text(encoding="utf-8-sig")
        for token in (
            "REJECT_WEIGHTED",
            "transformer_text_classifier",
            "tfidf_logistic_temperature_scaled",
            "REVIEW",
        ):
            if token not in decision_text:
                findings.append(
                    Finding(
                        "ERROR",
                        "decision_contract",
                        f"Expected decision token is missing: {token}",
                        decision_path.relative_to(repo).as_posix(),
                    )
                )

    provenance_path = (
        repo
        / "docs/artifacts/final_evaluation/transformer_training_provenance.json"
    )
    if provenance_path.is_file():
        try:
            provenance = json.loads(
                provenance_path.read_text(encoding="utf-8-sig")
            )
        except json.JSONDecodeError:
            return

        expected = {
            "checkpoint_selection_split": "calibration",
            "framework_model_selection_split": "selection",
            "final_evaluation_split": "test",
            "metric_for_best_model": "macro_f1",
            "load_best_model_at_end": True,
            "test_used_for_checkpoint_selection": False,
            "test_used_for_framework_selection": False,
            "test_used_for_reselection": False,
        }
        for key, expected_value in expected.items():
            actual = provenance.get(key)
            if actual != expected_value:
                findings.append(
                    Finding(
                        "ERROR",
                        "provenance_contract",
                        f"{key!r} is {actual!r}; expected {expected_value!r}.",
                        provenance_path.relative_to(repo).as_posix(),
                    )
                )

    weighted_path = (
        repo
        / "docs/artifacts/final_evaluation/weighted_challenger_decision.json"
    )
    if weighted_path.is_file():
        try:
            weighted = json.loads(weighted_path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError:
            return

        if "REJECT_WEIGHTED" not in set(all_string_values(weighted)):
            findings.append(
                Finding(
                    "ERROR",
                    "weighted_decision_contract",
                    "Weighted challenger artifact does not contain REJECT_WEIGHTED.",
                    weighted_path.relative_to(repo).as_posix(),
                )
            )


def check_readme(repo: Path, findings: list[Finding]) -> None:
    path = repo / "README.md"
    if not path.is_file():
        return

    text = path.read_text(encoding="utf-8-sig")
    required_tokens = (
        "<!-- public-release-overview:start -->",
        "docs/ARCHITECTURE.md",
        "docs/CLAIM_BOUNDARIES.md",
        "docs/FINAL_MODEL_DECISION.md",
        "docs/REPRODUCIBILITY.md",
    )
    for token in required_tokens:
        if token not in text:
            findings.append(
                Finding(
                    "ERROR",
                    "readme_contract",
                    f"README public overview is missing token: {token}",
                    "README.md",
                )
            )


def check_workflows(repo: Path, findings: list[Finding]) -> None:
    workflow_path = repo / ".github/workflows/release-contract.yml"
    if not workflow_path.is_file():
        return

    text = workflow_path.read_text(encoding="utf-8-sig")
    if "pull_request_target" in text:
        findings.append(
            Finding(
                "ERROR",
                "workflow_security",
                "pull_request_target is not allowed in the public release workflow.",
                workflow_path.relative_to(repo).as_posix(),
            )
        )

    for token in ("permissions:", "contents: read", "actions/checkout@", "actions/setup-python@"):
        if token not in text:
            findings.append(
                Finding(
                    "ERROR",
                    "workflow_contract",
                    f"Workflow is missing token: {token}",
                    workflow_path.relative_to(repo).as_posix(),
                )
            )


def check_git(repo: Path, findings: list[Finding], allow_dirty: bool) -> None:
    diff_check = run_git(repo, "diff", "--check")
    if diff_check.returncode != 0:
        findings.append(
            Finding(
                "ERROR",
                "git_diff_check",
                diff_check.stdout.strip() or diff_check.stderr.strip(),
            )
        )

    staged_check = run_git(repo, "diff", "--cached", "--check")
    if staged_check.returncode != 0:
        findings.append(
            Finding(
                "ERROR",
                "git_cached_diff_check",
                staged_check.stdout.strip() or staged_check.stderr.strip(),
            )
        )

    if not allow_dirty:
        status = run_git(repo, "status", "--short")
        if status.returncode == 0 and status.stdout.strip():
            findings.append(
                Finding(
                    "ERROR",
                    "git_clean",
                    "Working tree is not clean. Use --allow-dirty while reviewing a patch.",
                )
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the public GitHub release contract."
    )
    parser.add_argument("--repo", default=".", help="Repository root.")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow uncommitted changes while validating a release patch.",
    )
    parser.add_argument("--json-out", help="Optional JSON report path.")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    findings: list[Finding] = []

    if not (repo / ".git").is_dir():
        findings.append(
            Finding("ERROR", "git_repository", "The repository has no .git directory.")
        )

    tracked = tracked_files(repo)
    check_required_paths(repo, findings)
    check_tracked_hygiene(repo, tracked, findings)
    check_text_hygiene(repo, tracked, findings)
    check_structured_files(repo, tracked, findings)
    check_decision_contract(repo, findings)
    check_readme(repo, findings)
    check_workflows(repo, findings)
    check_git(repo, findings, args.allow_dirty)

    errors = [finding for finding in findings if finding.level == "ERROR"]
    warnings = [finding for finding in findings if finding.level == "WARN"]

    report = {
        "repository": str(repo),
        "tracked_file_count": len(tracked),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "status": "PASS" if not errors else "FAIL",
        "findings": [asdict(finding) for finding in findings],
    }

    if args.json_out:
        output_path = Path(args.json_out)
        if not output_path.is_absolute():
            output_path = repo / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    print("=" * 80)
    print("PUBLIC RELEASE CONTRACT")
    print("=" * 80)
    print(f"Repository:    {repo}")
    print(f"Tracked files: {len(tracked)}")
    print(f"Errors:        {len(errors)}")
    print(f"Warnings:      {len(warnings)}")
    print(f"Status:        {report['status']}")

    for finding in findings:
        location = f" [{finding.path}]" if finding.path else ""
        print(f"{finding.level}: {finding.check}{location}: {finding.message}")

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())

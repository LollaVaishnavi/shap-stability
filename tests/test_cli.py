from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shapstab import cli

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY = str(REPO_ROOT / "policy.toml")


def _write_stable_csv(path: Path, seed: int = 0) -> None:
    # Independent features with clearly separated coefficients: a ranking that should hold.
    rng = np.random.default_rng(seed)
    n = 1500
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    c = rng.standard_normal(n)
    y = 4.0 * a + 2.5 * b + 1.0 * c + 0.3 * rng.standard_normal(n)
    import pandas as pd

    pd.DataFrame({"a": a, "b": b, "c": c, "target": y}).to_csv(path, index=False)


def test_the_default_dataset_is_reported_unstable(capsys):
    code = cli.main(["--policy", POLICY, "--resamples", "200"])
    assert code == cli.EXIT_UNSTABLE
    assert "verdict: UNSTABLE" in capsys.readouterr().out


def test_quiet_prints_only_the_verdict_line(capsys):
    cli.main(["--policy", POLICY, "--resamples", "120", "--quiet"])
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1
    assert out[0].startswith("verdict:")


def test_a_clearly_separated_csv_is_reported_stable(tmp_path: Path):
    csv = tmp_path / "stable.csv"
    _write_stable_csv(csv)
    code = cli.main(["--policy", POLICY, "--csv", str(csv), "--target", "target"])
    assert code == cli.EXIT_STABLE


def test_csv_without_a_target_is_a_usage_error(tmp_path: Path, capsys):
    csv = tmp_path / "stable.csv"
    _write_stable_csv(csv)
    code = cli.main(["--policy", POLICY, "--csv", str(csv)])
    assert code == cli.EXIT_USAGE
    assert "target" in capsys.readouterr().err


def test_an_unknown_target_column_is_a_usage_error(tmp_path: Path):
    csv = tmp_path / "stable.csv"
    _write_stable_csv(csv)
    code = cli.main(["--policy", POLICY, "--csv", str(csv), "--target", "missing"])
    assert code == cli.EXIT_USAGE


def test_a_csv_with_no_feature_columns_is_a_usage_error(tmp_path: Path):
    csv = tmp_path / "only_target.csv"
    import pandas as pd

    pd.DataFrame({"target": [1.0, 2.0, 3.0]}).to_csv(csv, index=False)
    code = cli.main(["--policy", POLICY, "--csv", str(csv), "--target", "target"])
    assert code == cli.EXIT_USAGE


def test_a_bad_policy_file_is_a_usage_error(tmp_path: Path, capsys):
    bad = tmp_path / "bad.toml"
    bad.write_text("[resample]\nn_resamples = 1\nbase_seed = 1\n")
    code = cli.main(["--policy", str(bad)])
    assert code == cli.EXIT_USAGE
    assert "policy error" in capsys.readouterr().err


def test_an_impossible_rho_is_a_data_error(capsys):
    code = cli.main(["--policy", POLICY, "--rho", "1.5", "--resamples", "50"])
    assert code == cli.EXIT_USAGE
    assert "data error" in capsys.readouterr().err


def test_html_and_json_outputs_are_written(tmp_path: Path):
    html_path = tmp_path / "report.html"
    json_path = tmp_path / "report.json"
    cli.main(
        [
            "--policy",
            POLICY,
            "--resamples",
            "80",
            "--html",
            str(html_path),
            "--json",
            str(json_path),
        ]
    )
    assert "<span class=" in html_path.read_text()
    assert '"verdict"' in json_path.read_text()


def test_top_k_override_narrows_the_judged_depth(tmp_path: Path):
    json_path = tmp_path / "r.json"
    cli.main(["--policy", POLICY, "--resamples", "80", "--top-k", "3", "--json", str(json_path)])
    import json

    data = json.loads(json_path.read_text())
    assert data["top_k"] == 3


def test_a_lower_rho_still_runs_and_returns_a_verdict_code():
    code = cli.main(["--policy", POLICY, "--rho", "0.3", "--resamples", "80", "--quiet"])
    assert code in (cli.EXIT_STABLE, cli.EXIT_UNSTABLE, cli.EXIT_INCONCLUSIVE)


def test_help_lists_the_exit_codes(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    assert "exit codes" in capsys.readouterr().out

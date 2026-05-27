from pathlib import Path

import pytest

from bilayers_local import BilayersJob


def _config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config.yaml"


def test_bilayers_job_uses_yaml_defaults():
    pytest.importorskip("yaml")

    job = BilayersJob.from_cli([], config_path=_config_path())

    assert job.input_dir == Path("/bilayers/inputs")
    assert job.output_dir == Path("/bilayers/outputs")
    assert job.gt_dir == Path("/bilayers/gt")
    assert job.parameters.method == "ci_rl"
    assert job.parameters.iterations == "150"
    assert job.parameters.benchmark is False
    assert ".tif" in (job.suffixes or [])


def test_bilayers_job_cli_overrides_defaults():
    pytest.importorskip("yaml")

    job = BilayersJob.from_cli(
        [
            "--infolder",
            "C:/tmp/in",
            "--outfolder",
            "C:/tmp/out",
            "--method",
            "ci_sparse_hessian",
            "--benchmark",
            "True",
        ],
        config_path=_config_path(),
    )

    assert str(job.input_dir).replace("\\", "/") == "C:/tmp/in"
    assert str(job.output_dir).replace("\\", "/") == "C:/tmp/out"
    assert job.parameters.method == "ci_sparse_hessian"
    assert job.parameters.benchmark is True


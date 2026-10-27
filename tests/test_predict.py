"""predict.py 测试：端到端一致性、异常处理、校准器恢复。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from floodrisk.config import Config
from floodrisk.io import load_dataset
from floodrisk.predict import PredictionError, predict
from floodrisk.pipeline import run_training
from floodrisk.synthetic import write_synthetic_bundle


@pytest.fixture(scope="session")
def small_dyn_bundle(tmp_path_factory):
    out = tmp_path_factory.mktemp("syn_dyn_predict")
    return write_synthetic_bundle(out, mode="dynamic", n_grids=12, n_steps=48, seed=42)


@pytest.fixture(scope="session")
def trained_run(project_root, small_dyn_bundle, tmp_path_factory):
    """用合成数据快速训练，产出 run_dir 供 predict 测试。"""
    out = tmp_path_factory.mktemp("trained_run")
    cfg = Config.load(
        project_root / "configs" / "synthetic_dynamic.yaml",
        overrides={
            "data": {"input_path": str(small_dyn_bundle["data"])},
            "tuning": {"enabled": False},
            "output": {"dir": str(out), "save_model": True},
        },
    )
    manifest = run_training(cfg)
    return out / manifest["run_id"]


def test_predict_matches_training_output(trained_run, project_root, small_dyn_bundle):
    """端到端一致性：predict 同一批数据应与训练时的 predictions_val_test.csv 一致。"""
    result = predict(trained_run, small_dyn_bundle["data"])

    # 读取训练时的预测输出
    train_preds = pd.read_csv(trained_run / "predictions_val_test.csv")
    train_preds["timestamp"] = pd.to_datetime(train_preds["timestamp"])
    pred_probs = train_preds.set_index(["grid_id", "timestamp"])["prob_calibrated"]

    result_probs = result.df.set_index(["grid_id", "timestamp"])["prob_calibrated"]

    # 对齐比较（允许行序不同，按索引对齐）
    common = pred_probs.index.intersection(result_probs.index)
    assert len(common) > 0, "predict 结果与训练输出没有重叠的 (grid_id, timestamp)"

    diff = np.abs(pred_probs.loc[common].values - result_probs.loc[common].values)
    assert diff.max() < 1e-6, f"预测概率与训练输出不一致，最大差异 {diff.max()}"


def test_predict_without_label_column(trained_run, project_root, small_dyn_bundle):
    """无标签数据：去掉 flood_label 后应正常预测。"""
    df = pd.read_csv(small_dyn_bundle["data"])
    df = df.drop(columns=["flood_label"])
    no_label_path = small_dyn_bundle["data"].parent / "no_label.csv"
    df.to_csv(no_label_path, index=False)

    result = predict(trained_run, no_label_path)
    assert "prob_raw" in result.df.columns
    assert "prob_calibrated" in result.df.columns
    assert result.manifest["n_usable_rows"] > 0
    assert "flood_label" not in result.df.columns


def test_predict_missing_feature_column_raises(trained_run, project_root, small_dyn_bundle):
    """缺少必需特征列时应抛 DatasetError 或 PredictionError。"""
    df = pd.read_csv(small_dyn_bundle["data"])
    df = df.drop(columns=["rainfall"])
    bad_path = small_dyn_bundle["data"].parent / "missing_feature.csv"
    df.to_csv(bad_path, index=False)

    with pytest.raises(Exception, match="特征数量不匹配|rainfall|缺少必需字段"):
        predict(trained_run, bad_path)


def test_predict_calibrator_fallback(trained_run, small_dyn_bundle):
    """删除 calibrator.json 后应回退为未校准，并告警。"""
    cal_path = trained_run / "calibrator.json"
    backup = cal_path.read_text(encoding="utf-8")
    cal_path.unlink()

    result = predict(trained_run, small_dyn_bundle["data"])
    assert result.manifest["is_calibrated"] is False
    assert result.manifest["calibration_method"] == "none"
    assert "无校准器" in result.manifest["calibration_note"]

    # 恢复
    cal_path.write_text(backup, encoding="utf-8")


def test_predict_provenance_warning(trained_run, project_root, small_dyn_bundle):
    """合成训练 + 真实风格文件名预测 → provenance 告警。"""
    df = pd.read_csv(small_dyn_bundle["data"])
    fake_real_path = small_dyn_bundle["data"].parent / "hk_flood_2020_2024.csv"
    df.to_csv(fake_real_path, index=False)

    result = predict(trained_run, fake_real_path)
    assert result.manifest["provenance"] == "UNKNOWN"
    assert result.manifest["citable"] is False
    assert "PROVENANCE_UNKNOWN" in result.manifest["banner"]


def test_predict_output_columns(trained_run, small_dyn_bundle):
    """输出列必须包含 grid_id、timestamp、prob_raw、prob_calibrated、risk_level。"""
    result = predict(trained_run, small_dyn_bundle["data"])
    for col in ("grid_id", "timestamp", "prob_raw", "prob_calibrated", "risk_level"):
        assert col in result.df.columns, f"输出缺少列 {col}"
    assert result.df["risk_level"].isna().all()
    assert result.output_path.exists()


def test_predict_static_mode_not_supported(trained_run, project_root, small_dyn_bundle):
    """static 模式暂不支持独立预测。"""
    cfg_path = trained_run / "config_snapshot.yaml"
    cfg = Config.load(cfg_path)
    cfg._data["task"]["mode"] = "static"

    with pytest.raises(PredictionError, match="static"):
        predict(trained_run, small_dyn_bundle["data"], cfg=cfg)


def test_predict_run_dir_not_found(small_dyn_bundle):
    with pytest.raises(PredictionError, match="run_dir 不存在"):
        predict("/nonexistent/path", small_dyn_bundle["data"])


def test_predict_missing_model_json(small_dyn_bundle, tmp_path):
    bad_dir = tmp_path / "bad_run"
    bad_dir.mkdir()
    (bad_dir / "config_snapshot.yaml").write_text("task:\n  mode: dynamic\n", encoding="utf-8")
    with pytest.raises(PredictionError, match="model.json"):
        predict(bad_dir, small_dyn_bundle["data"])

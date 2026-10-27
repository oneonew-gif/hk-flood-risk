"""floodrisk —— 水浸风险预测建模框架（XGBoost + GIS）。

模块地图
--------
config    配置加载与校验（唯一默认值来源：configs/base.yaml）
schema    字段契约（规范名 ↔ 别名，两级映射；含预报类 kind=forecast）
io        读入、列映射、校验报告、provenance 标记
synthetic 合成数据（**仅程序自测**，坐标虚构，注入缺报）
splits    random / temporal / spatial_block / spacetime 切分 + 标签窗口 purge + 同源 CV
leakage   三道泄漏闸门（按时间轴）+ 单变量 AUC 扫描
features  按时间轴的 lag/rolling、预报窗口聚合、静态直通
train     XGBoost 训练、早停、不平衡处理、重要性
tuning    随机搜索（折与主切分同源）
evaluate  7 指标 + 混淆矩阵 + 阈值扫描 + 曲线
calibrate none / platt / isotonic + 可靠性曲线 + ECE
pipeline  端到端串联，产出 manifest 与诊断告警

典型用法
--------
>>> from floodrisk import Config, load_dataset, run_training
>>> cfg = Config.load("configs/synthetic_dynamic.yaml")
>>> manifest = run_training(cfg)
>>> manifest["metrics"]["test"]["pr_auc"]
"""

from __future__ import annotations

__version__ = "0.2.0"

from .calibrate import Calibrator, fit_calibrator
from .config import Config, ConfigError
from .evaluate import binary_metrics, threshold_sweep
from .features import FeatureError, FeatureSet, audit_built_features, build_features
from .io import Dataset, DatasetError, load_dataset
from .gis_export import GISExportError, export_geojson
from .leakage import LeakageError, audit_features, check_feature_names
from .pipeline import run_training
from .predict import PredictionError, PredictionResult, predict
from .splits import SplitError, SplitPlan, cv_splits, make_split_plan
from .train import TrainError, train_model
from .tuning import TuningResult, random_search

__all__ = [
    "__version__",
    "Calibrator",
    "fit_calibrator",
    "Config",
    "ConfigError",
    "binary_metrics",
    "threshold_sweep",
    "FeatureError",
    "FeatureSet",
    "audit_built_features",
    "build_features",
    "Dataset",
    "DatasetError",
    "load_dataset",
    "LeakageError",
    "audit_features",
    "check_feature_names",
    "run_training",
    "SplitError",
    "SplitPlan",
    "cv_splits",
    "make_split_plan",
    "TrainError",
    "train_model",
    "TuningResult",
    "random_search",
    "PredictionError",
    "PredictionResult",
    "predict",
    "GISExportError",
    "export_geojson",
]

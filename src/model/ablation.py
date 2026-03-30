"""
Ablation Study + 多模型比較

面試關鍵問題：
  「情緒特徵真的有幫助嗎？」→ Ablation Study 用對照組驗證
  「為什麼選 XGBoost？」→ 多模型比較提供量化依據

實驗設計：
  Feature Groups（消融實驗）:
    A. 只用技術指標（基準線）
    B. 只用 PTT 情緒特徵
    C. 技術指標 + PTT 情緒（主要模型）
    D. 技術指標 + PTT + 鉅亨新聞（完整版）

  Models（多模型比較）:
    - Logistic Regression（線性基準線）
    - Random Forest
    - XGBoost（主力模型）
    - LightGBM（若已安裝）
"""

import json
import logging
import warnings
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import FINAL_DIR

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ================================================================== #
#  特徵分組定義
# ================================================================== #

TECH_FEATURES = [
    "sma_5", "sma_20", "ema_12",
    "volume_change_pct",
    "price_vs_sma_5", "price_vs_sma_20",
    "intraday_volatility", "volatility_5d",
]

PTT_FEATURES = [
    "article_count", "avg_sentiment", "max_sentiment",
    "min_sentiment", "sentiment_std", "bullish_ratio", "push_net_avg",
    "sentiment_lag1", "sentiment_lag2", "sentiment_lag3",
    "sentiment_momentum", "article_count_ma3", "discussion_heat",
]

NEWS_FEATURES = [
    "news_article_count", "news_avg_sentiment",
    "news_max_sentiment", "news_min_sentiment",
    "news_sentiment_std", "news_bullish_ratio",
    "news_available",
    "news_sentiment_lag1", "news_sentiment_lag2",
    "sentiment_divergence",
]

FEATURE_GROUPS = {
    "A: 技術指標（基準線）": TECH_FEATURES,
    "B: PTT 情緒特徵": PTT_FEATURES,
    "C: 技術 + PTT 情緒": TECH_FEATURES + PTT_FEATURES,
    "D: 技術 + PTT + 新聞（完整）": TECH_FEATURES + PTT_FEATURES + NEWS_FEATURES,
}


def _get_model(name: str):
    """回傳對應模型實例"""
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=42), True  # needs scaling
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=100, random_state=42), False
    if name == "XGBoost":
        return XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=42, eval_metric="logloss", verbosity=0,
        ), False
    if name == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=42, verbose=-1,
        ), False
    raise ValueError(f"Unknown model: {name}")


def _evaluate(df: pd.DataFrame, feature_cols: list[str],
               model_name: str, n_splits: int = 5) -> dict:
    """
    用 Time Series Split 評估一個模型 + 特徵組合。
    回傳包含 avg_accuracy、avg_f1 等指標的 dict。
    """
    # 只保留 df 中實際存在的特徵欄位
    available = [c for c in feature_cols if c in df.columns]
    if not available:
        return {"error": "無可用特徵"}

    X = df[available].values
    y = df["label"].values

    model_cls, needs_scale = _get_model(model_name)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if needs_scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

        m = model_cls.__class__(**model_cls.get_params())
        m.fit(X_train, y_train)
        y_pred = m.predict(X_val)

        fold_results.append({
            "fold": fold + 1,
            "accuracy": round(accuracy_score(y_val, y_pred), 4),
            "f1": round(f1_score(y_val, y_pred, zero_division=0), 4),
            "precision": round(precision_score(y_val, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_val, y_pred, zero_division=0), 4),
        })

    avg_acc = round(float(np.mean([r["accuracy"] for r in fold_results])), 4)
    avg_f1  = round(float(np.mean([r["f1"] for r in fold_results])), 4)

    return {
        "avg_accuracy": avg_acc,
        "avg_f1": avg_f1,
        "n_features": len(available),
        "fold_results": fold_results,
    }


# ================================================================== #
#  Ablation Study
# ================================================================== #

def run_ablation(df: pd.DataFrame, n_splits: int = 5,
                  main_model: str = "XGBoost") -> dict:
    """
    Ablation Study：固定模型（XGBoost），切換特徵組合。
    目的：證明「加入情緒特徵是否帶來提升」
    """
    logger.info("=" * 50)
    logger.info("Ablation Study（固定 XGBoost，切換特徵組合）")
    logger.info("=" * 50)

    results = {}
    for group_name, feature_cols in FEATURE_GROUPS.items():
        logger.info(f"  評估: {group_name}")
        res = _evaluate(df, feature_cols, main_model, n_splits)
        results[group_name] = res
        logger.info(
            f"    Acc={res.get('avg_accuracy', 0):.2%}  "
            f"F1={res.get('avg_f1', 0):.2%}  "
            f"特徵數={res.get('n_features', 0)}"
        )

    return results


# ================================================================== #
#  多模型比較
# ================================================================== #

MODELS_TO_COMPARE = ["Logistic Regression", "Random Forest", "XGBoost"]

def run_model_comparison(df: pd.DataFrame, feature_cols: list[str],
                          n_splits: int = 5) -> dict:
    """
    多模型比較：固定特徵組合（技術 + PTT + 新聞），切換模型。
    目的：說明「為什麼選 XGBoost」
    """
    logger.info("=" * 50)
    logger.info("多模型比較（固定特徵組 C: 技術 + PTT 情緒）")
    logger.info("=" * 50)

    # 嘗試加入 LightGBM
    models = MODELS_TO_COMPARE.copy()
    try:
        import lightgbm  # noqa
        models.append("LightGBM")
    except ImportError:
        logger.info("LightGBM 未安裝，跳過")

    results = {}
    for model_name in models:
        logger.info(f"  評估: {model_name}")
        try:
            res = _evaluate(df, feature_cols, model_name, n_splits)
            results[model_name] = res
            logger.info(
                f"    Acc={res.get('avg_accuracy', 0):.2%}  "
                f"F1={res.get('avg_f1', 0):.2%}"
            )
        except Exception as e:
            logger.warning(f"  {model_name} 失敗: {e}")

    return results


# ================================================================== #
#  主流程
# ================================================================== #

def run_all(df: pd.DataFrame = None, n_splits: int = 5) -> dict:
    """執行全部實驗並儲存結果"""
    if df is None:
        filepath = FINAL_DIR / "features.csv"
        if not filepath.exists():
            logger.error(f"找不到 {filepath}，請先跑 main.py --stage features")
            return {}
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        df = df.dropna(subset=["label"])

    logger.info(f"資料集: {len(df)} 筆")

    ablation = run_ablation(df, n_splits=n_splits)
    comparison = run_model_comparison(
        df,
        feature_cols=TECH_FEATURES + PTT_FEATURES,
        n_splits=n_splits,
    )

    output = {
        "ablation": ablation,
        "model_comparison": comparison,
        "n_samples": len(df),
        "n_splits": n_splits,
    }

    out_path = FINAL_DIR / "ablation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"已儲存至 {out_path}")

    return output


# ================================================================== #
#  直接執行
# ================================================================== #
if __name__ == "__main__":
    print("=" * 60)
    print("Ablation Study + 多模型比較")
    print("=" * 60)
    results = run_all()

    print("\n--- Ablation Study 結果 ---")
    for group, res in results.get("ablation", {}).items():
        print(f"  {group:<30}  Acc={res['avg_accuracy']:.2%}  F1={res['avg_f1']:.2%}")

    print("\n--- 多模型比較結果 ---")
    for model, res in results.get("model_comparison", {}).items():
        print(f"  {model:<25}  Acc={res['avg_accuracy']:.2%}  F1={res['avg_f1']:.2%}")

# isort: skip_file
"""
模型訓練模組

重點設計：
1. 使用 XGBoost（處理表格資料比深度學習好）
2. Time Series Split（不能用隨機 CV，否則等於用未來預測過去）
3. 輸出特徵重要性（面試必問）
"""

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd
import json
import logging
from config import FINAL_DIR
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """XGBoost 模型訓練器，使用 Time Series Split"""

    def __init__(self):
        self.model = None
        self.feature_columns = []
        self.results = {}

    def train(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        n_splits: int = 5,
    ) -> dict:
        """
        訓練 XGBoost 分類模型。

        :param df: 特徵表（含 label 欄位）
        :param feature_columns: 要用的特徵欄位名稱
        :param n_splits: Time Series Split 的折數
        :return: 評估結果 dict
        """
        self.feature_columns = feature_columns

        X = df[feature_columns].values
        y = df["label"].values

        logger.info(f"資料集: {len(X)} 筆，{len(feature_columns)} 個特徵")
        logger.info(f"標籤分布: 漲={int((y == 1).sum())}, 跌={int((y == 0).sum())}")

        # --- Time Series Split ---
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_results = []

        print(f"\n--- Time Series Split ({n_splits} 折) ---")

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # 訓練 XGBoost
            model = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0,
            )
            model.fit(X_train, y_train)

            # 預測
            y_pred = model.predict(X_val)

            # 評估
            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)

            fold_results.append({
                "fold": fold + 1,
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                "accuracy": round(acc, 4),
                "f1": round(f1, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
            })

            print(f"  Fold {fold+1}: "
                  f"Train={len(train_idx)} Val={len(val_idx)} | "
                  f"Acc={acc:.2%} F1={f1:.2%} Prec={precision:.2%} Rec={recall:.2%}")

        # --- 用全部資料訓練最終模型 ---
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )
        self.model.fit(X, y)

        # --- 整體結果 ---
        avg_acc = np.mean([r["accuracy"] for r in fold_results])
        avg_f1 = np.mean([r["f1"] for r in fold_results])

        self.results = {
            "avg_accuracy": round(avg_acc, 4),
            "avg_f1": round(avg_f1, 4),
            "n_splits": n_splits,
            "n_samples": len(X),
            "n_features": len(feature_columns),
            "fold_results": fold_results,
        }

        print(f"\n--- 平均結果 ---")
        print(f"  Accuracy: {avg_acc:.2%}")
        print(f"  F1 Score: {avg_f1:.2%}")

        return self.results

    def get_feature_importance(self, top_n: int = 10) -> list[dict]:
        """
        取得特徵重要性排名。
        面試必問：「哪些特徵最重要？情緒特徵有沒有用？」
        """
        if self.model is None:
            logger.error("請先訓練模型")
            return []

        importances = self.model.feature_importances_
        feature_imp = sorted(
            zip(self.feature_columns, importances),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        print(f"\n--- 特徵重要性 Top {top_n} ---")
        for i, (name, imp) in enumerate(feature_imp[:top_n]):
            results.append({"rank": i + 1, "feature": name,
                           "importance": round(imp, 4)})
            bar = "█" * int(imp * 50)
            print(f"  {i+1:2d}. {name:<25s} {imp:.4f} {bar}")

        return results

    def save_results(self, filename: str = "model_results.json"):
        """儲存訓練結果（含特徵重要性）"""
        results_to_save = self.results.copy()

        # 將特徵重要性一起存入 JSON（Streamlit 直接讀取用）
        if self.model is not None:
            importances = self.model.feature_importances_
            feature_imp = sorted(
                zip(self.feature_columns, importances),
                key=lambda x: x[1],
                reverse=True,
            )
            results_to_save["feature_importance"] = [
                {"feature": name, "importance": round(float(imp), 4)}
                for name, imp in feature_imp
            ]

        filepath = FINAL_DIR / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results_to_save, f, indent=2, ensure_ascii=False)
        logger.info(f"已儲存訓練結果至 {filepath}")


# ================================================================== #
#  測試區塊（用模擬資料）
# ================================================================== #
if __name__ == "__main__":
    print("=" * 60)
    print("模型訓練測試（模擬資料）")
    print("=" * 60)

    # 建立模擬特徵表
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2025-01-01", periods=n, freq="B")

    df = pd.DataFrame({
        "avg_sentiment": np.random.uniform(-0.5, 0.5, n),
        "sentiment_std": np.random.uniform(0, 0.5, n),
        "bullish_ratio": np.random.uniform(0.3, 0.7, n),
        "article_count": np.random.randint(0, 20, n),
        "push_net_avg": np.random.uniform(-5, 15, n),
        "sentiment_lag1": np.random.uniform(-0.5, 0.5, n),
        "sentiment_lag2": np.random.uniform(-0.5, 0.5, n),
        "sentiment_momentum": np.random.uniform(-0.3, 0.3, n),
        "sma_5": 900 + np.random.randn(n) * 10,
        "sma_20": 900 + np.random.randn(n) * 5,
        "ema_12": 900 + np.random.randn(n) * 8,
        "volume_change_pct": np.random.randn(n) * 20,
        "price_vs_sma_5": np.random.randn(n) * 2,
        "intraday_volatility": np.random.uniform(0.5, 3, n),
        "volatility_5d": np.random.uniform(0.5, 2, n),
        "label": np.random.choice([0.0, 1.0], n),
    }, index=dates)

    feature_cols = [col for col in df.columns if col != "label"]

    trainer = ModelTrainer()
    results = trainer.train(df, feature_cols, n_splits=5)
    trainer.get_feature_importance(top_n=10)
    trainer.save_results()

    print(f"\n✓ 結果已儲存至 data/final/model_results.json")

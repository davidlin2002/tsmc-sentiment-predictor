# isort: skip_file
"""
特徵工程模組

將情緒資料與股價資料合併，產出模型可用的特徵表。
特徵分三大類：情緒特徵、技術指標、標籤（隔天漲跌）
"""

import numpy as np
import pandas as pd
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

# 可調參數（之後可以搬到 config.py）
SMA_WINDOWS = [5, 20]
EMA_WINDOW = 12
SENTIMENT_LAGS = [1, 2, 3]
LABEL_THRESHOLD = 0.5  # ±0.5% 漲跌門檻


class FeatureEngineer:
    """合併情緒與股價資料，產出最終特徵表"""

    def build_features(
        self,
        price_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        news_sentiment_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        主流程：合併 → 技術指標 → 情緒衍生特徵 → 標籤

        :param price_df: 股價資料
        :param sentiment_df: PTT 每日情緒彙整
        :param news_sentiment_df: 鉅亨網每日情緒彙整（選填，None 則跳過新聞特徵）
        """
        logger.info("開始建立特徵...")

        # Step 1: 統一 index 格式
        price_df = price_df.copy()
        price_df.index = pd.to_datetime(price_df.index)
        price_df.index.name = "trade_date"

        sentiment_df = sentiment_df.copy()
        sentiment_df.index = pd.to_datetime(sentiment_df.index)
        sentiment_df.index.name = "trade_date"

        # Step 2: 合併 PTT 情緒（以股價為主，left join）
        df = price_df.join(sentiment_df, how="left")

        # 沒有對應情緒的交易日填 0（代表那天沒有相關 PTT 文章）
        sentiment_cols = [
            "article_count", "avg_sentiment", "max_sentiment",
            "min_sentiment", "sentiment_std", "bullish_ratio", "push_net_avg",
        ]
        for col in sentiment_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # Step 2b: 合併新聞情緒（選填）
        if news_sentiment_df is not None and not news_sentiment_df.empty:
            news_df = news_sentiment_df.copy()
            news_df.index = pd.to_datetime(news_df.index)
            news_df.index.name = "trade_date"

            news_cols_map = {
                "article_count": "news_article_count",
                "avg_sentiment": "news_avg_sentiment",
                "max_sentiment": "news_max_sentiment",
                "min_sentiment": "news_min_sentiment",
                "sentiment_std": "news_sentiment_std",
                "bullish_ratio": "news_bullish_ratio",
            }
            news_renamed = news_df.rename(columns=news_cols_map)
            cols_to_join = [c for c in news_cols_map.values() if c in news_renamed.columns]
            df = df.join(news_renamed[cols_to_join], how="left")

            # news_available: 1 = 該天有新聞，0 = 無新聞（與「新聞情緒中性」區分）
            df["news_available"] = df["news_avg_sentiment"].notna().astype(float)
            for col in cols_to_join:
                df[col] = df[col].fillna(0)

            logger.info(f"已合併新聞情緒，覆蓋 {int(df['news_available'].sum())} 個交易日")

        # Step 3: 技術指標
        df = self._add_technical_features(df)

        # Step 4: PTT 情緒衍生特徵
        df = self._add_sentiment_features(df)

        # Step 4b: 新聞情緒衍生特徵
        df = self._add_news_sentiment_features(df)

        # Step 5: 標籤（隔天漲跌）
        df = self._add_label(df)

        # Step 6: 移除前幾列（移動平均會產生 NaN）
        max_window = max(SMA_WINDOWS + [EMA_WINDOW] + SENTIMENT_LAGS)
        df = df.iloc[max_window:]

        # 移除標籤為 NaN 的列
        df = df.dropna(subset=["label"])

        logger.info(f"特徵建立完成: {len(df)} 筆資料，{len(df.columns)} 個欄位")
        return df

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """取得特徵欄位名稱（排除標籤和原始價格）"""
        exclude = [
            "label", "open", "high", "low", "close",
            "volume", "change_pct", "next_change_pct",
        ]
        return [col for col in df.columns if col not in exclude]

    def save_features(self, df: pd.DataFrame, filename: str = "features.csv"):
        """儲存最終特徵表"""
        filepath = FINAL_DIR / filename
        df.to_csv(filepath)
        logger.info(f"已儲存特徵表至 {filepath}")

    # ================================================================ #
    #  技術指標
    # ================================================================ #

    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """加入股價技術指標"""

        # 移動平均線 (SMA)
        for window in SMA_WINDOWS:
            df[f"sma_{window}"] = df["close"].rolling(window=window).mean()

        # 指數移動平均 (EMA)
        df[f"ema_{EMA_WINDOW}"] = df["close"].ewm(
            span=EMA_WINDOW, adjust=False).mean()

        # 成交量變化率
        df["volume_change_pct"] = df["volume"].pct_change() * 100

        # 價格偏離 SMA 的程度
        for window in SMA_WINDOWS:
            df[f"price_vs_sma_{window}"] = (
                (df["close"] - df[f"sma_{window}"]) / df[f"sma_{window}"] * 100
            )

        # 日內波動率
        df["intraday_volatility"] = (
            df["high"] - df["low"]) / df["close"] * 100

        # 過去 5 天波動率
        df["volatility_5d"] = df["change_pct"].rolling(window=5).std()

        return df

    # ================================================================ #
    #  情緒衍生特徵
    # ================================================================ #

    def _add_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """加入情緒衍生特徵"""

        if "avg_sentiment" not in df.columns:
            logger.warning("缺少 avg_sentiment，跳過情緒特徵")
            return df

        # 情緒落後指標：前 1/2/3 天的情緒
        for lag in SENTIMENT_LAGS:
            df[f"sentiment_lag{lag}"] = df["avg_sentiment"].shift(lag)

        # 情緒動能：近 3 天 vs 近 7 天
        df["sentiment_momentum"] = (
            df["avg_sentiment"].rolling(3).mean() -
            df["avg_sentiment"].rolling(7).mean()
        )

        # 討論熱度指標
        if "article_count" in df.columns:
            df["article_count_ma3"] = df["article_count"].rolling(3).mean()
            df["discussion_heat"] = (
                df["article_count"] / df["article_count_ma3"].replace(0, 1)
            )

        return df

    # ================================================================ #
    #  新聞情緒衍生特徵
    # ================================================================ #

    def _add_news_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        加入鉅亨網新聞情緒衍生特徵。
        若 news_avg_sentiment 欄位不存在（未提供新聞資料）則直接跳過。
        """
        if "news_avg_sentiment" not in df.columns:
            return df

        # 新聞落後指標
        for lag in [1, 2]:
            df[f"news_sentiment_lag{lag}"] = df["news_avg_sentiment"].shift(lag)

        # 情緒分歧：PTT（社群）vs 鉅亨網（媒體）
        # 正值 = 鄉民比媒體樂觀；負值 = 媒體比鄉民樂觀
        # 大絕對值 = 兩者意見分歧，可能是雜訊或反轉訊號
        if "avg_sentiment" in df.columns:
            df["sentiment_divergence"] = df["avg_sentiment"] - df["news_avg_sentiment"]

        return df

    # ================================================================ #
    #  標籤
    # ================================================================ #

    def _add_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        建立標籤：隔天漲跌（二元分類）

        設計決策：
        - ±0.5% 門檻，中間的當噪音移除
        - 漲 = 1，跌 = 0
        """
        df["next_change_pct"] = df["change_pct"].shift(-1)

        df["label"] = np.nan
        df.loc[df["next_change_pct"] > LABEL_THRESHOLD, "label"] = 1.0
        df.loc[df["next_change_pct"] < -LABEL_THRESHOLD, "label"] = 0.0

        return df


# ================================================================== #
#  測試區塊（用模擬資料）
# ================================================================== #
if __name__ == "__main__":
    print("=" * 60)
    print("特徵工程測試（模擬資料）")
    print("=" * 60)

    # 建立模擬股價
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=60, freq="B")
    close = 900 + np.cumsum(np.random.randn(60) * 5)

    price_df = pd.DataFrame({
        "open": close + np.random.randn(60) * 2,
        "high": close + abs(np.random.randn(60) * 5),
        "low": close - abs(np.random.randn(60) * 5),
        "close": close,
        "volume": np.random.randint(30_000_000, 60_000_000, 60),
        "change_pct": np.random.randn(60) * 1.5,
    }, index=dates)

    # 建立模擬每日情緒
    sentiment_df = pd.DataFrame({
        "article_count": np.random.randint(0, 20, 60),
        "avg_sentiment": np.random.uniform(-0.5, 0.5, 60),
        "max_sentiment": np.random.uniform(0, 1, 60),
        "min_sentiment": np.random.uniform(-1, 0, 60),
        "sentiment_std": np.random.uniform(0, 0.5, 60),
        "bullish_ratio": np.random.uniform(0.3, 0.7, 60),
        "push_net_avg": np.random.uniform(-5, 15, 60),
    }, index=dates)

    engineer = FeatureEngineer()
    features = engineer.build_features(price_df, sentiment_df)

    print(f"\n✓ 最終特徵表: {features.shape[0]} 筆 × {features.shape[1]} 欄")
    print(f"\n特徵欄位 ({len(engineer.get_feature_columns(features))} 個):")
    for col in engineer.get_feature_columns(features):
        print(f"  - {col}")
    print(f"\n標籤分布:")
    print(f"  漲(1): {int((features['label'] == 1).sum())} 筆")
    print(f"  跌(0): {int((features['label'] == 0).sum())} 筆")

    engineer.save_features(features)
    print(f"\n✓ 已儲存至 data/final/features.csv")

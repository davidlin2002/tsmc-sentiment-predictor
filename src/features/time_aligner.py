# isort: skip_file
"""
時間對齊模組 — 將 PTT 文章對應到正確的交易日

核心邏輯（T+1）：今天的輿論 → 影響明天的股價
這解決了 Gemini 完全沒處理的時間對齊問題。
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
from config import PROCESSED_DIR
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class TimeAligner:
    """將情緒資料對齊到交易日"""

    def __init__(self):
        self.trading_dates = set()

    def set_trading_dates_from_price(self, price_df: pd.DataFrame):
        """
        從股價 DataFrame 設定交易日。
        用實際股價資料比「跳過週末」更準確，因為還能處理國定假日。
        """
        self.trading_dates = set()
        for d in price_df.index:
            if isinstance(d, str):
                d = pd.Timestamp(d)
            self.trading_dates.add(d.date() if hasattr(d, 'date') else d)
        logger.info(f"已設定 {len(self.trading_dates)} 個交易日")

    def get_target_trade_date(self, article_datetime: datetime) -> pd.Timestamp:
        """
        將文章日期對應到「受影響的交易日」（T+1）。
        從文章隔天開始，找到下一個交易日。
        """
        if article_datetime is None:
            return None

        article_date = article_datetime.date() if hasattr(
            article_datetime, 'date') else article_datetime
        candidate = article_date + timedelta(days=1)

        # 如果有實際交易日資料，用它來找
        if self.trading_dates:
            for _ in range(10):  # 最多找 10 天（涵蓋長假）
                if candidate in self.trading_dates:
                    return pd.Timestamp(candidate)
                candidate += timedelta(days=1)
            return None

        # 沒有交易日資料時，簡單跳過週末
        while candidate.weekday() >= 5:
            candidate += timedelta(days=1)
        return pd.Timestamp(candidate)

    def aggregate_daily_sentiment(self, articles: list[dict]) -> pd.DataFrame:
        """
        將文章按交易日聚合，產出每日情緒摘要。

        輸出欄位：
        - article_count: 文章數量（討論熱度）
        - avg_sentiment: 平均情緒分數
        - max_sentiment / min_sentiment: 情緒極值
        - sentiment_std: 情緒分歧度（多空爭議程度）
        - bullish_ratio: 看多文章佔比
        - push_net_avg: 平均淨推文數（推 - 噓）
        """
        records = []

        for article in articles:
            # 解析 datetime
            dt = article.get("datetime")
            if isinstance(dt, str):
                try:
                    dt = datetime.fromisoformat(dt)
                except ValueError:
                    continue
            if dt is None:
                continue

            score = article.get("sentiment_score")
            if score is None:
                continue

            # 對齊到交易日
            trade_date = self.get_target_trade_date(dt)
            if trade_date is None:
                continue

            # 計算淨推文數
            push_count = article.get("push_count", {})
            if isinstance(push_count, str):
                try:
                    push_count = json.loads(push_count)
                except json.JSONDecodeError:
                    push_count = {}

            push_net = push_count.get("推", 0) - push_count.get("噓", 0)

            records.append({
                "trade_date": trade_date,
                "sentiment_score": float(score),
                "push_net": push_net,
                "label": article.get("sentiment_label", "neutral"),
            })

        if not records:
            logger.warning("沒有可聚合的文章")
            return pd.DataFrame()

        df = pd.DataFrame(records)

        # 按交易日聚合
        grouped = df.groupby("trade_date")
        daily = pd.DataFrame({
            "article_count": grouped.size(),
            "avg_sentiment": grouped["sentiment_score"].mean(),
            "max_sentiment": grouped["sentiment_score"].max(),
            "min_sentiment": grouped["sentiment_score"].min(),
            "sentiment_std": grouped["sentiment_score"].std().fillna(0),
            "bullish_ratio": grouped.apply(
                lambda x: (x["label"].isin(["bullish", "very_bullish"])).mean()
            ),
            "push_net_avg": grouped["push_net"].mean(),
        })

        daily.index.name = "trade_date"
        daily = daily.sort_index()

        # 四捨五入
        for col in daily.select_dtypes(include=[np.floating]).columns:
            daily[col] = daily[col].round(4)

        logger.info(f"聚合完成: {len(daily)} 個交易日的情緒資料")
        return daily

    def save_daily_sentiment(self, df: pd.DataFrame, filename: str = "daily_sentiment.csv"):
        """儲存每日情緒聚合結果"""
        filepath = PROCESSED_DIR / filename
        df.to_csv(filepath)
        logger.info(f"已儲存每日情緒至 {filepath}")


# ================================================================== #
#  測試區塊
# ================================================================== #
if __name__ == "__main__":
    aligner = TimeAligner()

    print("=" * 60)
    print("時間對齊測試")
    print("=" * 60)

    test_cases = [
        ("週一文章", datetime(2025, 3, 24, 10, 30)),
        ("週三文章", datetime(2025, 3, 26, 20, 0)),
        ("週五文章", datetime(2025, 3, 28, 15, 0)),
        ("週六文章", datetime(2025, 3, 29, 12, 0)),
        ("週日文章", datetime(2025, 3, 30, 9, 0)),
    ]

    weekday_names = ["一", "二", "三", "四", "五", "六", "日"]

    for desc, dt in test_cases:
        target = aligner.get_target_trade_date(dt)
        src_day = weekday_names[dt.weekday()]
        tgt_day = weekday_names[target.weekday()] if target else "N/A"
        print(f"  {desc} (週{src_day} {dt.strftime('%m/%d')}) → 交易日: 週{tgt_day} {target.strftime('%m/%d') if target else 'N/A'}")

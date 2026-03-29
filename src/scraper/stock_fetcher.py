# isort: skip_file
"""
台股股價抓取模組

改進 Gemini 版本：
1. yfinance 為主，TWSE 官方 API 為備案
2. 自動計算漲跌幅百分比（change_pct）
3. 欄位名稱統一 snake_case
"""

import time
import logging
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from config import STOCK_TICKER, STOCK_CODE, STOCK_NAME, RAW_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class StockFetcher:
    """台股股價資料抓取器"""

    def fetch_price(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        抓取股價資料，yfinance 失敗時自動切換 TWSE API。

        :param start_date: 起始日期 (YYYY-MM-DD)
        :param end_date: 結束日期 (YYYY-MM-DD)
        :return: DataFrame，columns = [open, high, low, close, volume, change_pct]
        """
        df = self._fetch_yfinance(start_date, end_date)

        if df.empty:
            logger.warning("yfinance 抓取失敗，切換至 TWSE API...")
            df = self._fetch_twse(start_date, end_date)

        if df.empty:
            logger.error("所有資料來源都抓取失敗")
            return pd.DataFrame()

        df = self._post_process(df)
        return df

    def save_price(self, df: pd.DataFrame, filename: str = "stock_price.csv"):
        """儲存股價至 CSV"""
        filepath = RAW_DIR / filename
        df.to_csv(filepath)
        logger.info(f"已儲存 {len(df)} 筆股價至 {filepath}")

    # ================================================================== #
    #  資料來源
    # ================================================================== #

    def _fetch_yfinance(self, start_date: str, end_date: str) -> pd.DataFrame:
        """使用 yfinance 抓取"""
        try:
            logger.info(
                f"使用 yfinance 抓取 {STOCK_TICKER} ({start_date} ~ {end_date})...")

            stock = yf.Ticker(STOCK_TICKER)
            df = stock.history(start=start_date, end=end_date)

            if df.empty:
                logger.warning(f"yfinance 回傳空資料")
                return pd.DataFrame()

            # 移除時區
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            # 統一欄位名稱為 snake_case
            df = df.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            })
            df = df[["open", "high", "low", "close", "volume"]]
            df.index.name = "date"

            logger.info(f"yfinance 成功抓取 {len(df)} 筆資料")
            return df

        except Exception as e:
            logger.error(f"yfinance 錯誤: {e}")
            return pd.DataFrame()

    def _fetch_twse(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        使用 TWSE 官方 API 抓取（備案）。
        注意：每次只回傳一個月，需按月循環。每 5 秒最多 3 個 request。
        """
        try:
            import requests

            logger.info(
                f"使用 TWSE API 抓取 {STOCK_CODE} ({start_date} ~ {end_date})...")

            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            all_data = []
            current = start.replace(day=1)

            while current <= end:
                date_param = current.strftime("%Y%m%d")
                url = (
                    f"https://www.twse.com.tw/exchangeReport/STOCK_DAY"
                    f"?response=json&date={date_param}&stockNo={STOCK_CODE}"
                )

                try:
                    resp = requests.get(url, timeout=10)
                    data = resp.json()

                    if data.get("stat") == "OK" and data.get("data"):
                        for row in data["data"]:
                            parsed = self._parse_twse_row(row)
                            if parsed["date"] is not None:
                                all_data.append(parsed)
                except Exception as e:
                    logger.warning(f"TWSE API 錯誤 ({date_param}): {e}")

                # 下一個月
                if current.month == 12:
                    current = current.replace(year=current.year + 1, month=1)
                else:
                    current = current.replace(month=current.month + 1)

                time.sleep(2)  # 遵守 rate limit

            if not all_data:
                return pd.DataFrame()

            df = pd.DataFrame(all_data)
            df = df.set_index("date").sort_index()
            df = df.loc[start_date:end_date]

            logger.info(f"TWSE API 成功抓取 {len(df)} 筆資料")
            return df

        except Exception as e:
            logger.error(f"TWSE API 錯誤: {e}")
            return pd.DataFrame()

    @staticmethod
    def _parse_twse_row(row: list) -> dict:
        """解析 TWSE API 單筆資料（民國年 → 西元年）"""
        try:
            date_parts = row[0].strip().split("/")
            year = int(date_parts[0]) + 1911
            month = int(date_parts[1])
            day = int(date_parts[2])
            date = datetime(year, month, day)

            def clean_num(s):
                return float(s.replace(",", "")) if s.replace(",", "").replace(".", "").replace("-", "").strip() else 0.0

            return {
                "date": date,
                "open": clean_num(row[3]),
                "high": clean_num(row[4]),
                "low": clean_num(row[5]),
                "close": clean_num(row[6]),
                "volume": int(clean_num(row[1])),
            }
        except (IndexError, ValueError) as e:
            logger.warning(f"解析 TWSE 資料列失敗: {row}, 錯誤: {e}")
            return {"date": None, "open": 0, "high": 0, "low": 0, "close": 0, "volume": 0}

    @staticmethod
    def _post_process(df: pd.DataFrame) -> pd.DataFrame:
        """計算漲跌幅、排序、清理"""
        if df.empty:
            return df

        df = df.sort_index()
        df["change_pct"] = df["close"].pct_change() * 100
        df = df.iloc[1:]  # 移除第一筆（沒有前一天）

        for col in ["open", "high", "low", "close", "change_pct"]:
            if col in df.columns:
                df[col] = df[col].round(2)

        return df


# ================================================================== #
#  測試區塊
# ================================================================== #
if __name__ == "__main__":
    fetcher = StockFetcher()

    # 抓最近 3 個月的資料做測試
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

    print("=" * 60)
    print(f"抓取 {STOCK_NAME} ({STOCK_TICKER}) 股價")
    print(f"期間: {start_date} ~ {end_date}")
    print("=" * 60)

    df = fetcher.fetch_price(start_date, end_date)

    if not df.empty:
        print(f"\n✓ 成功抓取 {len(df)} 筆資料")
        print(
            f"  日期範圍: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"  收盤價範圍: {df['close'].min():.2f} ~ {df['close'].max():.2f}")
        print(f"  平均成交量: {df['volume'].mean():,.0f}")
        print(f"\n--- 最近 5 筆 ---")
        print(df.tail())

        fetcher.save_price(df)
        print(f"\n✓ 已儲存至 data/raw/stock_price.csv")
    else:
        print("✗ 抓取失敗！")

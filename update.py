# isort: skip_file
"""
每日更新腳本 — 補抓新資料，不動舊資料

執行順序：
  1. 補抓今日股價（與舊資料合併去重）
  2. 補抓最新 PTT 文章（跳過已有的）
  3. 補抓最新鉅亨新聞（跳過已有的）
  4. 對新文章跑 LLM 情緒分析（舊的走快取）
  5. 重建特徵工程 + 重訓模型
"""

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import PROCESSED_DIR, RAW_DIR, STOCK_NAME

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ================================================================== #
#  Step 1: 股價更新
# ================================================================== #
def update_stock_price():
    from src.scraper.stock_fetcher import StockFetcher

    logger.info("--- Step 1: 更新股價 ---")
    fetcher = StockFetcher()
    price_path = RAW_DIR / "stock_price.csv"

    if price_path.exists():
        old = pd.read_csv(price_path, index_col=0, parse_dates=True)
        last_date = old.index.max()
        # 從上次最後一天往前 5 天重抓（保留緩衝，確保 change_pct 計算正確）
        start = (last_date - timedelta(days=5)).strftime("%Y-%m-%d")
        logger.info(f"現有資料最新: {last_date.date()}，從 {start} 補抓")
    else:
        old = pd.DataFrame()
        start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

    # yfinance 的 end 是不含當天，所以要傳明天才能包含今天
    end = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    new_price = fetcher.fetch_price(start, end)

    if new_price.empty:
        logger.warning("股價抓取失敗，跳過")
        return old if not old.empty else pd.DataFrame()

    if not old.empty:
        combined = pd.concat([old, new_price])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = new_price

    combined.to_csv(price_path)
    latest = combined.index.max().date()
    logger.info(f"股價已更新: 共 {len(combined)} 筆，最新交易日 {latest}")
    return combined


# ================================================================== #
#  Step 2: PTT 文章更新
# ================================================================== #
def update_ptt_articles() -> list[dict]:
    from src.scraper.ptt_scraper import PTTScraper

    logger.info("--- Step 2: 更新 PTT 文章 ---")
    ptt_path = RAW_DIR / "ptt_articles.jsonl"
    existing = _load_jsonl(ptt_path)
    existing_ids = {a.get("article_id") for a in existing}
    logger.info(f"現有 PTT 文章: {len(existing)} 篇")

    scraper = PTTScraper()
    fetched = scraper.collect_tsmc_articles(max_search_pages=2, max_detail_articles=50)
    new_articles = [a for a in fetched if a.get("article_id") not in existing_ids]
    logger.info(f"新 PTT 文章: {len(new_articles)} 篇")

    if new_articles:
        all_articles = existing + new_articles
        scraper.save_articles(all_articles)
        logger.info(f"ptt_articles.jsonl 已更新: 共 {len(all_articles)} 篇")
    return new_articles


# ================================================================== #
#  Step 3: 鉅亨新聞更新
# ================================================================== #
def update_cnyes_articles() -> list[dict]:
    from src.scraper.cnyes_scraper import CnyesScraper

    logger.info("--- Step 3: 更新鉅亨新聞 ---")
    cnyes_path = RAW_DIR / "cnyes_articles.jsonl"
    existing = _load_jsonl(cnyes_path)
    existing_ids = {a.get("article_id") for a in existing}
    logger.info(f"現有鉅亨新聞: {len(existing)} 篇")

    # 只抓最近 7 天（每日更新不需要抓太遠）
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

    try:
        scraper = CnyesScraper()
        fetched = scraper.fetch_articles(start, end, max_articles=200)
        new_articles = [a for a in fetched if a.get("article_id") not in existing_ids]
        logger.info(f"新鉅亨新聞: {len(new_articles)} 篇")

        if new_articles:
            all_articles = existing + new_articles
            scraper.save_articles(all_articles)
            logger.info(f"cnyes_articles.jsonl 已更新: 共 {len(all_articles)} 篇")
        return new_articles
    except Exception as e:
        logger.warning(f"鉅亨新聞更新失敗（跳過）: {e}")
        return []


# ================================================================== #
#  Step 4: 情緒分析（只處理新文章）
# ================================================================== #
def update_sentiment(new_ptt: list[dict], new_cnyes: list[dict]):
    from src.sentiment.llm_analyzer import SentimentAnalyzer

    logger.info("--- Step 4: 情緒分析 ---")
    if not new_ptt and not new_cnyes:
        logger.info("無新文章，跳過情緒分析")
        return

    analyzer = SentimentAnalyzer()

    # PTT 新文章
    if new_ptt:
        logger.info(f"分析 {len(new_ptt)} 篇新 PTT 文章...")
        enriched_new = analyzer.batch_analyze(new_ptt, source="ptt")

        ptt_sent_path = PROCESSED_DIR / "ptt_with_sentiment.jsonl"
        old_enriched = _load_jsonl(ptt_sent_path)
        analyzer.save_results(old_enriched + enriched_new)
        logger.info(f"PTT 情緒標註: 共 {len(old_enriched) + len(enriched_new)} 篇")

    # 鉅亨新文章
    if new_cnyes:
        logger.info(f"分析 {len(new_cnyes)} 篇新鉅亨新聞...")
        enriched_news = analyzer.batch_analyze(new_cnyes, source="news")

        cnyes_sent_path = PROCESSED_DIR / "cnyes_with_sentiment.jsonl"
        old_cnyes = _load_jsonl(cnyes_sent_path)
        analyzer.save_results(old_cnyes + enriched_news, filename="cnyes_with_sentiment.jsonl")
        logger.info(f"鉅亨情緒標註: 共 {len(old_cnyes) + len(enriched_news)} 篇")


# ================================================================== #
#  Step 5: 重建特徵 + 重訓模型
# ================================================================== #
def rebuild_features_and_model():
    from src.features.time_aligner import TimeAligner
    from src.features.feature_engineer import FeatureEngineer
    from src.model.trainer import ModelTrainer

    logger.info("--- Step 5: 重建特徵 + 重訓模型 ---")

    price_df = pd.read_csv(RAW_DIR / "stock_price.csv", index_col=0, parse_dates=True)

    ptt_articles = _load_jsonl(PROCESSED_DIR / "ptt_with_sentiment.jsonl")
    cnyes_articles = _load_jsonl(PROCESSED_DIR / "cnyes_with_sentiment.jsonl")

    aligner = TimeAligner()
    aligner.set_trading_dates_from_price(price_df)

    daily_sentiment = aligner.aggregate_daily_sentiment(ptt_articles)
    aligner.save_daily_sentiment(daily_sentiment)

    news_daily = None
    if cnyes_articles:
        news_daily = aligner.aggregate_daily_sentiment(cnyes_articles)
        aligner.save_daily_sentiment(news_daily, filename="daily_news_sentiment.csv")

    engineer = FeatureEngineer()
    features = engineer.build_features(price_df, daily_sentiment, news_daily)
    engineer.save_features(features)

    feature_cols = engineer.get_feature_columns(features)
    trainer = ModelTrainer()
    trainer.train(features, feature_cols, n_splits=5)
    trainer.get_feature_importance(top_n=10)
    trainer.save_results()

    logger.info(f"模型重訓完成: {len(features)} 筆資料，{len(feature_cols)} 個特徵")


# ================================================================== #
#  主流程
# ================================================================== #
def update():
    start_time = datetime.now()
    logger.info(f"{'='*50}")
    logger.info(f"{STOCK_NAME} 每日更新開始 {start_time.strftime('%Y-%m-%d %H:%M')}")
    logger.info(f"{'='*50}")

    update_stock_price()
    new_ptt = update_ptt_articles()
    new_cnyes = update_cnyes_articles()
    update_sentiment(new_ptt, new_cnyes)
    rebuild_features_and_model()

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"{'='*50}")
    logger.info(f"每日更新完成！耗時 {elapsed:.0f} 秒")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    update()

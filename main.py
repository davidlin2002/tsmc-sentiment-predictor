# isort: skip_file
"""
主流程腳本 — 一鍵執行完整 pipeline

Usage:
    python main.py --test           # 測試模式（少量資料）
    python main.py --stage scrape   # 只跑爬蟲
    python main.py --stage sentiment # 只跑情緒分析
    python main.py --stage features  # 只跑特徵工程
    python main.py --stage model     # 只跑模型訓練
    python main.py                   # 執行全部
"""

import pandas as pd
from datetime import datetime, timedelta
import logging
import json
import argparse
from config import RAW_DIR, PROCESSED_DIR, FINAL_DIR, STOCK_NAME
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def stage_scrape(test_mode=False):
    """Stage 1: 資料獲取"""
    from src.scraper.ptt_scraper import PTTScraper
    from src.scraper.stock_fetcher import StockFetcher
    from src.scraper.cnyes_scraper import CnyesScraper

    logger.info("=" * 50)
    logger.info("Stage 1a: PTT 文章爬取")
    logger.info("=" * 50)

    scraper = PTTScraper()
    articles = scraper.collect_tsmc_articles(
        max_search_pages=2 if test_mode else 20,
        max_detail_articles=5 if test_mode else 500,
    )
    scraper.save_articles(articles)

    logger.info("=" * 50)
    logger.info("Stage 1b: 股價抓取（自動匹配文章時間範圍）")
    logger.info("=" * 50)

    fetcher = StockFetcher()

    if test_mode or not articles:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    else:
        # 從文章日期自動決定股價範圍
        article_dates = []
        for a in articles:
            dt = a.get("datetime")
            if dt:
                if isinstance(dt, datetime):
                    article_dates.append(dt)
                elif isinstance(dt, str):
                    try:
                        article_dates.append(datetime.fromisoformat(dt))
                    except ValueError:
                        pass

        if article_dates:
            earliest = min(article_dates)
            latest = max(article_dates)
            # 往前多抓 30 天（技術指標需要歷史資料算移動平均）
            start_date = (earliest - timedelta(days=30)).strftime("%Y-%m-%d")
            end_date = (latest + timedelta(days=5)).strftime("%Y-%m-%d")
            logger.info(
                f"文章時間範圍: {earliest.strftime('%Y-%m-%d')} ~ {latest.strftime('%Y-%m-%d')}")
            logger.info(f"股價抓取範圍: {start_date} ~ {end_date}（前後各留緩衝）")
        else:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=90)
                          ).strftime("%Y-%m-%d")

    price_df = fetcher.fetch_price(start_date, end_date)
    if not price_df.empty:
        fetcher.save_price(price_df)

    logger.info("=" * 50)
    logger.info("Stage 1c: 鉅亨網新聞爬取")
    logger.info("=" * 50)

    cnyes_articles = []
    try:
        cnyes_scraper = CnyesScraper()
        cnyes_articles = cnyes_scraper.fetch_articles(
            start_date=start_date,
            end_date=end_date,
            max_articles=20 if test_mode else 1000,
        )
        if cnyes_articles:
            cnyes_scraper.save_articles(cnyes_articles)
    except Exception as e:
        logger.warning(f"鉅亨網爬取失敗（跳過）: {e}")

    return articles, price_df, cnyes_articles


def stage_sentiment(articles=None, cnyes_articles=None):
    """Stage 2: LLM 情緒分析（PTT + 鉅亨網）"""
    from src.sentiment.llm_analyzer import SentimentAnalyzer

    logger.info("=" * 50)
    logger.info("Stage 2: LLM 情緒分析")
    logger.info("=" * 50)

    # 從檔案讀取（如果沒傳入）
    if articles is None:
        filepath = RAW_DIR / "ptt_articles.jsonl"
        if not filepath.exists():
            logger.error(f"找不到 {filepath}，請先跑 stage_scrape")
            return [], []
        articles = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                articles.append(json.loads(line))

    analyzer = SentimentAnalyzer()

    # PTT 情緒分析
    enriched = analyzer.batch_analyze(articles, source="ptt")
    analyzer.save_results(enriched)

    # 鉅亨網情緒分析
    cnyes_enriched = []
    if cnyes_articles is None:
        cnyes_path = RAW_DIR / "cnyes_articles.jsonl"
        if cnyes_path.exists():
            cnyes_articles = []
            with open(cnyes_path, "r", encoding="utf-8") as f:
                for line in f:
                    cnyes_articles.append(json.loads(line))

    if cnyes_articles:
        logger.info(f"開始分析 {len(cnyes_articles)} 篇鉅亨新聞...")
        cnyes_enriched = analyzer.batch_analyze(cnyes_articles, source="news")
        analyzer.save_results(cnyes_enriched, filename="cnyes_with_sentiment.jsonl")

    stats = analyzer.get_cache_stats()
    logger.info(f"快取中有 {stats['cached_articles']} 篇文章（含新聞）")

    return enriched, cnyes_enriched


def stage_features(enriched_articles=None, price_df=None, cnyes_enriched=None):
    """Stage 3: 時間對齊 + 特徵工程（PTT + 鉅亨網）"""
    from src.features.time_aligner import TimeAligner
    from src.features.feature_engineer import FeatureEngineer

    logger.info("=" * 50)
    logger.info("Stage 3: 時間對齊 + 特徵工程")
    logger.info("=" * 50)

    # 從檔案讀取 PTT 資料
    if enriched_articles is None:
        filepath = PROCESSED_DIR / "ptt_with_sentiment.jsonl"
        if not filepath.exists():
            logger.error(f"找不到 {filepath}")
            return None
        enriched_articles = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                enriched_articles.append(json.loads(line))

    if price_df is None:
        filepath = RAW_DIR / "stock_price.csv"
        if not filepath.exists():
            logger.error(f"找不到 {filepath}")
            return None
        price_df = pd.read_csv(filepath, index_col=0, parse_dates=True)

    aligner = TimeAligner()
    aligner.set_trading_dates_from_price(price_df)

    # PTT 時間對齊
    daily_sentiment = aligner.aggregate_daily_sentiment(enriched_articles)
    aligner.save_daily_sentiment(daily_sentiment)

    # 鉅亨網時間對齊（選填）
    news_daily_sentiment = None
    if cnyes_enriched is None:
        cnyes_path = PROCESSED_DIR / "cnyes_with_sentiment.jsonl"
        if cnyes_path.exists():
            cnyes_enriched = []
            with open(cnyes_path, "r", encoding="utf-8") as f:
                for line in f:
                    cnyes_enriched.append(json.loads(line))

    if cnyes_enriched:
        news_daily_sentiment = aligner.aggregate_daily_sentiment(cnyes_enriched)
        aligner.save_daily_sentiment(news_daily_sentiment, filename="daily_news_sentiment.csv")
        logger.info(f"新聞每日情緒: {len(news_daily_sentiment)} 個交易日")

    engineer = FeatureEngineer()
    features = engineer.build_features(price_df, daily_sentiment, news_daily_sentiment)
    engineer.save_features(features)

    return features


def stage_model(features=None):
    """Stage 4: 模型訓練"""
    from src.features.feature_engineer import FeatureEngineer
    from src.model.trainer import ModelTrainer

    logger.info("=" * 50)
    logger.info("Stage 4: 模型訓練")
    logger.info("=" * 50)

    if features is None:
        filepath = FINAL_DIR / "features.csv"
        if not filepath.exists():
            logger.error(f"找不到 {filepath}")
            return None
        features = pd.read_csv(filepath, index_col=0, parse_dates=True)

    engineer = FeatureEngineer()
    feature_cols = engineer.get_feature_columns(features)

    trainer = ModelTrainer()
    results = trainer.train(features, feature_cols, n_splits=5)
    trainer.get_feature_importance(top_n=10)
    trainer.save_results()

    return results


def main():
    parser = argparse.ArgumentParser(description=f"{STOCK_NAME} 情緒分析 Pipeline")
    parser.add_argument("--stage", choices=["scrape", "sentiment", "features", "model", "all"],
                        default="all")
    parser.add_argument("--test", action="store_true", help="測試模式（少量資料）")
    args = parser.parse_args()

    start_time = datetime.now()
    logger.info(f"Pipeline 開始 — 階段: {args.stage}, 測試模式: {args.test}")

    articles, price_df, cnyes_articles = None, None, None
    enriched, cnyes_enriched = None, None
    features = None

    if args.stage in ("scrape", "all"):
        articles, price_df, cnyes_articles = stage_scrape(test_mode=args.test)

    if args.stage in ("sentiment", "all"):
        enriched, cnyes_enriched = stage_sentiment(articles, cnyes_articles)

    if args.stage in ("features", "all"):
        features = stage_features(enriched, price_df, cnyes_enriched)

    if args.stage in ("model", "all"):
        stage_model(features)

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Pipeline 完成！耗時 {elapsed:.1f} 秒")


if __name__ == "__main__":
    main()

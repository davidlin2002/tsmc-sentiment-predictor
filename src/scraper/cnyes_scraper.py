"""
鉅亨網 (Anue) 新聞爬蟲

使用鉅亨網公開 API 抓取台積電相關財經新聞。
回傳格式與 PTTScraper 相容，可直接送入 SentimentAnalyzer。
"""

import json
import html
import logging
import re
import time
from datetime import datetime
from pathlib import Path
import sys

import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    CNYES_API_URL,
    CNYES_PAGE_SIZE,
    CNYES_REQUEST_DELAY,
    RAW_DIR,
    TSMC_KEYWORDS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class CnyesScraper:
    """鉅亨網財經新聞爬蟲，使用 time-walking cursor 分頁"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
        })

    # ================================================================ #
    #  公開介面
    # ================================================================ #

    def fetch_articles(
        self,
        start_date: str,
        end_date: str,
        max_articles: int = 1000,
    ) -> list[dict]:
        """
        抓取指定日期範圍內的台積電相關新聞。
        API 使用 page-based 分頁，從第 1 頁往後走，
        直到文章日期早於 start_date 為止。

        :param start_date: 開始日期 YYYY-MM-DD
        :param end_date: 結束日期 YYYY-MM-DD
        :param max_articles: 最多抓幾篇（測試模式傳小值）
        :return: 文章 list，欄位同 PTTScraper
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59
        )

        logger.info(f"鉅亨網新聞抓取: {start_date} ~ {end_date}，最多 {max_articles} 篇")

        articles = []
        seen_ids = set()
        page = 1

        while len(articles) < max_articles:
            raw_items, pagination = self._fetch_page(page)
            if not raw_items:
                break

            new_count = 0
            oldest_dt_on_page = end_dt  # 追蹤本頁最舊日期

            for item in raw_items:
                news_id = item.get("newsId") or item.get("id")
                if not news_id:
                    continue

                article_id = f"cnyes_{news_id}"
                if article_id in seen_ids:
                    continue

                parsed = self._parse_article(item)
                if parsed is None:
                    continue

                article_dt = parsed["datetime"]
                if article_dt < oldest_dt_on_page:
                    oldest_dt_on_page = article_dt

                # 日期範圍過濾
                if article_dt > end_dt:
                    continue
                if article_dt < start_dt:
                    continue

                if not self._is_tsmc_related(parsed["title"], parsed["content"]):
                    continue

                seen_ids.add(article_id)
                articles.append(parsed)
                new_count += 1

                if len(articles) >= max_articles:
                    break

            logger.info(
                f"  第 {page} 頁（共 {pagination.get('last_page', '?')} 頁）"
                f"，過濾後新增 {new_count} 篇，累計 {len(articles)} 篇"
                f"，本頁最舊: {oldest_dt_on_page.strftime('%Y-%m-%d')}"
            )

            # 本頁最舊文章已早於 start_date，不必再往下翻
            if oldest_dt_on_page < start_dt:
                logger.info("已到達目標日期範圍起點，停止翻頁")
                break

            # 沒有下一頁
            if not pagination.get("next_page_url"):
                break

            if len(articles) >= max_articles:
                break

            page += 1
            time.sleep(CNYES_REQUEST_DELAY)

        logger.info(f"鉅亨網爬取完成，共 {len(articles)} 篇台積電相關新聞")
        return articles

    def save_articles(
        self,
        articles: list[dict],
        filename: str = "cnyes_articles.jsonl",
    ) -> None:
        """儲存文章至 JSONL 檔，datetime 序列化為 ISO 格式"""
        filepath = RAW_DIR / filename
        with open(filepath, "w", encoding="utf-8") as f:
            for article in articles:
                record = article.copy()
                if isinstance(record.get("datetime"), datetime):
                    record["datetime"] = record["datetime"].isoformat()
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"已儲存 {len(articles)} 篇新聞至 {filepath}")

    # ================================================================ #
    #  私有方法
    # ================================================================ #

    def _fetch_page(self, page: int = 1) -> tuple[list[dict], dict]:
        """
        呼叫鉅亨網 API 取得單頁新聞列表。

        實際回應結構：
        {
          "data": {
            "total": 2149, "per_page": 30, "current_page": 1,
            "last_page": 72,
            "next_page_url": "/media/api/v1/newslist/category/tw_stock?page=2",
            "data": [ {...article...}, ... ]
          }
        }

        :return: (items list, pagination dict)，失敗回傳 ([], {})
        """
        params = {"page": page}
        try:
            resp = self.session.get(CNYES_API_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            if not isinstance(data, dict):
                logger.warning(f"API 回應格式異常（非 dict）: {str(data)[:200]}")
                return [], {}

            # 實際結構: {"items": {"total":..., "data": [...articles...]}, "statusCode": 200}
            pagination = data.get("items", {})
            if not isinstance(pagination, dict):
                logger.warning(f"API pagination 格式異常: {str(pagination)[:200]}")
                return [], {}

            # 文章列表在分頁物件內的 "data" 鍵
            items = pagination.get("data", [])
            if not isinstance(items, list):
                logger.warning(f"API items 不是 list: {str(items)[:200]}")
                return [], pagination

            return items, pagination

        except requests.RequestException as e:
            logger.error(f"API 請求失敗: {e}")
            return [], {}
        except (ValueError, KeyError) as e:
            logger.error(f"API 回應解析失敗: {e}")
            return [], {}

    def _parse_article(self, item: dict) -> dict | None:
        """
        將 Anue API item 轉換成標準文章格式。
        失敗回傳 None。
        """
        try:
            news_id = item.get("newsId") or item.get("id")
            if not news_id:
                return None

            title = item.get("title", "").strip()
            if not title:
                return None

            # 時間：publishAt 是 Unix timestamp（整數）
            publish_ts = item.get("publishAt", 0)
            if publish_ts:
                article_dt = datetime.fromtimestamp(publish_ts)
            else:
                return None

            # 內容：可能是 HTML，清洗成純文字
            raw_content = (
                item.get("content")
                or item.get("summary")
                or item.get("intro")
                or ""
            )
            content = self._clean_html(raw_content)

            # URL
            url = (
                item.get("url")
                or f"https://news.cnyes.com/news/id/{news_id}"
            )

            return {
                "article_id": f"cnyes_{news_id}",
                "title": title,
                "datetime": article_dt,
                "content": content,
                "url": url,
                "source": "cnyes",
                # PTT 相容欄位（新聞沒有這些，填預設值）
                "author": item.get("source", "鉅亨網"),
                "push_count": {},  # 新聞沒有推/噓
                "category": "新聞",
            }
        except Exception as e:
            logger.debug(f"解析 item 失敗: {e}, item={str(item)[:100]}")
            return None

    def _is_tsmc_related(self, title: str, content: str = "") -> bool:
        """
        判斷文章是否與台積電相關。
        同時檢查標題與內文前 300 字，提高召回率。
        """
        check_text = title + " " + content[:300]
        return any(kw in check_text for kw in TSMC_KEYWORDS)

    @staticmethod
    def _clean_html(raw: str) -> str:
        """
        將 HTML 轉成純文字。
        鉅亨網 API 回傳雙層編碼（&lt;p&gt;），需先 unescape 再去標籤。

        流程：
        1. 還原 HTML entities（&lt; → <）
        2. 移除 <script> / <style> 區塊
        3. 移除所有 HTML 標籤
        4. 再次還原 entities（處理 &amp; 等）
        5. 合併多餘空白
        """
        if not raw:
            return ""
        # Step 1: 先 unescape，把 &lt;p&gt; 還原成 <p>
        text = html.unescape(raw)
        # Step 2: 移除 script / style 區塊
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE)
        # Step 3: 移除所有 HTML 標籤
        text = re.sub(r"<[^>]+>", " ", text)
        # Step 4: 再次 unescape 處理殘留 entities
        text = html.unescape(text)
        # Step 5: 合併多餘空白
        text = re.sub(r"\s+", " ", text).strip()
        return text


# ================================================================== #
#  獨立測試區塊
# ================================================================== #
if __name__ == "__main__":
    print("=" * 60)
    print("鉅亨網爬蟲測試")
    print("=" * 60)

    scraper = CnyesScraper()

    # 測試抓取近 30 天資料
    from datetime import timedelta
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    articles = scraper.fetch_articles(start, end, max_articles=20)

    print(f"\n抓到 {len(articles)} 篇")
    for a in articles[:3]:
        print(f"\n  ID   : {a['article_id']}")
        print(f"  標題 : {a['title']}")
        print(f"  時間 : {a['datetime']}")
        print(f"  內容 : {a['content'][:100]}...")

    if articles:
        scraper.save_articles(articles)
        print(f"\n[OK] 已儲存至 data/raw/cnyes_articles.jsonl")

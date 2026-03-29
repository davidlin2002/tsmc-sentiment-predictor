# isort: skip_file
"""
PTT Stock 版爬蟲

改進 Gemini 版本：
1. 使用 PTT 搜尋 API 精準找文章
2. 進入文章內頁抓完整 datetime（含年份）
3. 分開計算推/噓/箭頭數
4. 解析文章分類 tag（[標的]、[新聞] 等）
5. 從 URL 提取 article_id 做去重和快取 key
"""

from bs4 import BeautifulSoup
import requests
from typing import Optional
from datetime import datetime
import logging
import time
import copy
import json
import re
from config import (
    PTT_BASE_URL, PTT_SEARCH_URL, PTT_BOARD,
    PTT_COOKIES, PTT_REQUEST_DELAY, PTT_MAX_RETRIES,
    TSMC_KEYWORDS, PTT_CATEGORIES, RAW_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class PTTScraper:
    """PTT Stock 版爬蟲，支援搜尋 API 和逐頁爬取兩種模式"""

    def __init__(self):
        self.session = requests.Session()
        self.session.cookies.update(PTT_COOKIES)
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
        })

    # ================================================================== #
    #  公開介面
    # ================================================================== #

    def search_articles(self, keyword: str, max_pages: int = 10) -> list[dict]:
        """
        使用 PTT 搜尋 API 抓取包含特定關鍵字的文章列表。
        比逐頁翻找高效很多——Gemini 版本沒用到這個功能。

        :param keyword: 搜尋關鍵字（例如 "台積電" 或 "2330"）
        :param max_pages: 最多搜尋幾頁結果
        :return: 文章 metadata 列表
        """
        articles = []
        url = PTT_SEARCH_URL
        params = {"q": keyword}

        for page_num in range(max_pages):
            logger.info(f"搜尋 '{keyword}' 第 {page_num + 1} 頁...")
            soup = self._fetch_page(url, params=params)
            if soup is None:
                break

            page_articles = self._parse_article_list(soup)
            if not page_articles:
                logger.info("沒有更多搜尋結果")
                break

            articles.extend(page_articles)

            # 找「上一頁」連結
            prev_link = soup.find("a", string="‹ 上頁")
            if prev_link and prev_link.get("href"):
                url = PTT_BASE_URL + prev_link["href"]
                params = None  # URL 已包含 query string
            else:
                break

            time.sleep(PTT_REQUEST_DELAY)

        logger.info(f"搜尋 '{keyword}' 完成，共找到 {len(articles)} 篇文章")
        return articles

    def fetch_article_detail(self, article_url: str) -> Optional[dict]:
        """
        進入文章內頁，抓取完整資訊：
        - 完整 datetime（含年份）
        - 文章內文
        - 推/噓/箭頭分開計算
        """
        soup = self._fetch_page(article_url)
        if soup is None:
            return None

        try:
            main_content = soup.find("div", id="main-content")
            if not main_content:
                return None

            # --- 解析 meta 資訊 ---
            meta_values = main_content.find_all(
                "span", class_="article-meta-value")
            if len(meta_values) < 4:
                logger.warning(f"文章 meta 不完整: {article_url}")
                return None

            author = meta_values[0].text.strip()
            title = meta_values[2].text.strip()
            date_str = meta_values[3].text.strip()

            # 解析完整日期，例如 "Thu Mar 27 14:23:00 2025"
            article_datetime = self._parse_ptt_datetime(date_str)

            # --- 解析文章內文 ---
            content = self._extract_content(main_content)

            # --- 解析推噓文 ---
            push_tags = main_content.find_all("div", class_="push")
            push_count = {"推": 0, "噓": 0, "→": 0}
            for tag in push_tags:
                push_type_elem = tag.find("span", class_="push-tag")
                if push_type_elem:
                    push_type = push_type_elem.text.strip()
                    if push_type in push_count:
                        push_count[push_type] += 1

            # --- 提取 article_id ---
            article_id = self._extract_article_id(article_url)

            # --- 解析文章分類 ---
            category = self._extract_category(title)

            return {
                "article_id": article_id,
                "title": title,
                "author": author,
                "datetime": article_datetime,
                "content": content,
                "push_count": push_count,
                "url": article_url,
                "category": category,
            }

        except Exception as e:
            logger.error(f"解析文章失敗 {article_url}: {e}")
            return None

    def collect_tsmc_articles(
        self,
        max_search_pages: int = 5,
        max_detail_articles: int = 200,
    ) -> list[dict]:
        """
        完整的台積電文章收集流程：
        1. 用多個關鍵字搜尋
        2. 去重
        3. 逐篇抓取詳細內容
        """
        # Step 1: 多關鍵字搜尋
        all_articles = {}  # 用 article_id 去重
        search_keywords = ["台積電", "2330"]

        for keyword in search_keywords:
            results = self.search_articles(keyword, max_pages=max_search_pages)
            for article in results:
                aid = article.get("article_id")
                if aid and aid not in all_articles:
                    all_articles[aid] = article

        logger.info(f"去重後共 {len(all_articles)} 篇不重複文章")

        # Step 2: 過濾真正相關的文章
        relevant_articles = {}
        for aid, article in all_articles.items():
            title = article.get("title", "")
            if self._is_tsmc_related(title):
                relevant_articles[aid] = article

        logger.info(f"過濾後剩 {len(relevant_articles)} 篇台積電相關文章")

        # Step 3: 逐篇抓詳細內容
        articles_to_fetch = list(relevant_articles.values())[
            :max_detail_articles]
        detailed_articles = []

        for i, article in enumerate(articles_to_fetch):
            url = article.get("url")
            if not url:
                continue

            logger.info(
                f"抓取文章詳情 ({i+1}/{len(articles_to_fetch)}): {article.get('title', '')[:30]}...")
            detail = self.fetch_article_detail(url)

            if detail:
                detailed_articles.append(detail)

            time.sleep(PTT_REQUEST_DELAY)

        logger.info(f"成功抓取 {len(detailed_articles)} 篇文章詳情")
        return detailed_articles

    def save_articles(self, articles: list[dict], filename: str = "ptt_articles.jsonl"):
        """儲存文章至 JSONL 格式"""
        filepath = RAW_DIR / filename
        with open(filepath, "w", encoding="utf-8") as f:
            for article in articles:
                article_copy = article.copy()
                if isinstance(article_copy.get("datetime"), datetime):
                    article_copy["datetime"] = article_copy["datetime"].isoformat()
                f.write(json.dumps(article_copy, ensure_ascii=False) + "\n")
        logger.info(f"已儲存 {len(articles)} 篇文章至 {filepath}")

    # ================================================================== #
    #  內部方法
    # ================================================================== #

    def _fetch_page(self, url: str, params: dict = None) -> Optional[BeautifulSoup]:
        """帶重試機制的頁面請求"""
        for attempt in range(PTT_MAX_RETRIES):
            try:
                resp = self.session.get(url, params=params, timeout=10)
                resp.raise_for_status()
                return BeautifulSoup(resp.text, "lxml")
            except requests.RequestException as e:
                logger.warning(f"請求失敗 (第 {attempt+1} 次): {e}")
                if attempt < PTT_MAX_RETRIES - 1:
                    time.sleep(PTT_REQUEST_DELAY * (attempt + 1))
        return None

    def _parse_article_list(self, soup: BeautifulSoup) -> list[dict]:
        """從列表頁解析文章 metadata"""
        articles = []
        divs = soup.find_all("div", class_="r-ent")

        for div in divs:
            try:
                title_elem = div.find("a")
                if not title_elem:
                    continue  # 文章已被刪除

                title = title_elem.text.strip()
                href = title_elem.get("href", "")
                url = PTT_BASE_URL + href if href else ""
                article_id = self._extract_article_id(url)

                # 推文數（列表頁只有淨推文數）
                nrec_elem = div.find("div", class_="nrec")
                nrec_text = nrec_elem.text.strip() if nrec_elem else ""
                nrec = self._parse_nrec(nrec_text)

                # 日期（列表頁只有 mm/dd）
                date_elem = div.find("div", class_="date")
                list_date = date_elem.text.strip() if date_elem else ""

                # 作者
                author_elem = div.find("div", class_="author")
                author = author_elem.text.strip() if author_elem else ""

                articles.append({
                    "article_id": article_id,
                    "title": title,
                    "url": url,
                    "author": author,
                    "list_date": list_date,
                    "nrec": nrec,
                })
            except Exception as e:
                logger.debug(f"解析列表項目失敗: {e}")
                continue

        return articles

    def _extract_content(self, main_content) -> str:
        """從文章 main-content div 中提取純文字內容"""
        content_div = copy.copy(main_content)

        # 移除 meta 區塊
        for meta in content_div.find_all("div", class_="article-metaline"):
            meta.decompose()
        for meta in content_div.find_all("div", class_="article-metaline-right"):
            meta.decompose()

        # 移除推文區塊
        for push in content_div.find_all("div", class_="push"):
            push.decompose()

        # 取得文字，用 --\n 分割取正文
        text = content_div.get_text()
        if "\n--\n" in text:
            text = text.split("\n--\n")[0]

        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text

    @staticmethod
    def _parse_ptt_datetime(date_str: str) -> Optional[datetime]:
        """解析 PTT 文章內頁的日期格式，例如 'Thu Mar 27 14:23:00 2025'"""
        try:
            return datetime.strptime(date_str, "%a %b %d %H:%M:%S %Y")
        except ValueError:
            try:
                return datetime.strptime(date_str.strip(), "%a %b  %d %H:%M:%S %Y")
            except ValueError:
                logger.warning(f"無法解析日期: '{date_str}'")
                return None

    @staticmethod
    def _extract_article_id(url: str) -> str:
        """從 URL 提取 article_id，例如 M.1711234567.A.B12"""
        match = re.search(r"/(M\.\d+\.A\.[A-Z0-9]+)\.html", url)
        return match.group(1) if match else ""

    @staticmethod
    def _extract_category(title: str) -> str:
        """從標題提取文章分類，例如 '[標的] 2330 台積電 多' → '標的'"""
        match = re.match(r"\[([^\]]+)\]", title)
        if match:
            cat = match.group(1).strip()
            for standard_cat in PTT_CATEGORIES:
                if standard_cat in cat:
                    return standard_cat
            return cat
        return "其他"

    @staticmethod
    def _parse_nrec(nrec_text: str) -> int:
        """解析推文數：數字、'爆'、'X1'~'XX' 等"""
        if not nrec_text:
            return 0
        if nrec_text == "爆":
            return 100
        if nrec_text.startswith("X"):
            try:
                return -10 * int(nrec_text[1:]) if len(nrec_text) > 1 else -10
            except ValueError:
                return -100
        try:
            return int(nrec_text)
        except ValueError:
            return 0

    @staticmethod
    def _is_tsmc_related(title: str) -> bool:
        """判斷標題是否與台積電相關（使用同義詞表）"""
        title_upper = title.upper()
        for keyword in TSMC_KEYWORDS:
            if keyword.upper() in title_upper:
                return True
        return False


# ================================================================== #
#  測試區塊
# ================================================================== #
# ================================================================== #
#  測試區塊
# ================================================================== #
if __name__ == "__main__":
    scraper = PTTScraper()

    # --- 測試 1: 搜尋 API ---
    print("=" * 60)
    print("測試 1: 用搜尋 API 找台積電文章")
    print("=" * 60)

    results = scraper.search_articles("台積電", max_pages=1)
    print(f"\n搜尋到 {len(results)} 篇文章：")
    for r in results[:5]:
        print(f"  [{r['list_date']}] {r['title']}")

    # --- 測試 2: 抓一篇文章的詳細內容 ---
    if results:
        print("\n" + "=" * 60)
        print("測試 2: 抓取第一篇文章的完整內容")
        print("=" * 60)

        first_url = results[0]["url"]
        detail = scraper.fetch_article_detail(first_url)

        if detail:
            dt = detail["datetime"]
            dt_str = dt.strftime("%Y-%m-%d %H:%M") if dt else "N/A"
            push = detail["push_count"]

            print(f"\n  標題: {detail['title']}")
            print(f"  作者: {detail['author']}")
            print(f"  完整時間: {dt_str}")
            print(f"  分類: {detail['category']}")
            print(f"  推/噓/→: {push['推']}/{push['噓']}/{push['→']}")
            print(f"  內文前 100 字:")
            print(f"  {detail['content'][:100]}...")
        else:
            print("  抓取失敗")

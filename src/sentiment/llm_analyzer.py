# isort: skip_file
"""
LLM 情緒分析模組（使用 Google Gemini API）

面試亮點：
1. SQLite 快取 — article_id 為 key，避免重複呼叫浪費錢
2. PTT 鄉民黑話理解 — Prompt 專門處理「丸子」「咕嚕咕嚕」等用語
3. 結構化輸出 — 確保 LLM 回傳可解析的 JSON
4. 模擬模式 — 沒有 API Key 也能跑通 pipeline
"""

from typing import Optional
from datetime import datetime
import logging
import sqlite3
import json
from config import GEMINI_API_KEY, STOCK_NAME, PROCESSED_DIR
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# === Prompt 模板 ===
SENTIMENT_PROMPT = f"""你是一位專精台灣股市的資深財經分析師，非常熟悉 PTT 股票版的鄉民用語。

你的任務是分析每一則 PTT 文章對 {STOCK_NAME} 的情緒影響。

## PTT 鄉民常見用語對照：
- 「丸子」「丸了」= 完了（利空）
- 「咕嚕咕嚕」= 股價下沉（利空）
- 「噴」「噴出」「火箭」= 股價急漲（利多）
- 「抄底」「撿鑽石」= 認為已到低點，準備買進（偏多）
- 「韭菜」「被割」= 散戶虧錢（利空情緒）
- 「多軍」= 看多的人；「空軍」= 看空的人
- 「春燕」= 景氣回春訊號（利多）
- 「護國神山」「GG」= 台積電
- 「外資提款機」= 外資賣超（利空）
- 「融資斷頭」= 強制平倉（極度利空）
- 「法說會」= 法人說明會（重要事件）

## 回傳格式：
只回傳一個 JSON 物件，不要有任何其他文字：

{{"score": <float>, "label": "<string>", "reasoning": "<string>"}}

- score: -1.0（極度利空）到 1.0（極度利多），精確到小數點後二位
- label: very_bearish / bearish / neutral / bullish / very_bullish
- reasoning: 一句話簡述判斷理由（繁體中文）

分數對應：
- very_bearish: -1.0 ~ -0.6
- bearish: -0.6 ~ -0.2
- neutral: -0.2 ~ 0.2
- bullish: 0.2 ~ 0.6
- very_bullish: 0.6 ~ 1.0
"""

# 新聞專用 prompt（移除 PTT 俚語詞彙表，財經新聞用正式中文）
NEWS_SENTIMENT_PROMPT = f"""你是一位專精台灣股市的資深財經分析師。

你的任務是分析每一則財經新聞對 {STOCK_NAME} 的情緒影響。

## 回傳格式：
只回傳一個 JSON 物件，不要有任何其他文字：

{{"score": <float>, "label": "<string>", "reasoning": "<string>"}}

- score: -1.0（極度利空）到 1.0（極度利多），精確到小數點後二位
- label: very_bearish / bearish / neutral / bullish / very_bullish
- reasoning: 一句話簡述判斷理由（繁體中文）

分數對應：
- very_bearish: -1.0 ~ -0.6
- bearish: -0.6 ~ -0.2
- neutral: -0.2 ~ 0.2
- bullish: 0.2 ~ 0.6
- very_bullish: 0.6 ~ 1.0
"""


class SentimentAnalyzer:
    """使用 Gemini LLM 進行情緒分析，內建 SQLite 快取"""

    def __init__(self, cache_db: str = "sentiment_cache.db"):
        self.cache_db_path = PROCESSED_DIR / cache_db
        self._init_cache_db()
        self.model = None  # lazy init

    # ================================================================== #
    #  公開介面
    # ================================================================== #

    def analyze(self, article_id: str, title: str, content: str = "") -> Optional[dict]:
        """
        分析單篇文章的情緒。先查快取，沒有才呼叫 LLM。
        """
        # Step 1: 查快取
        cached = self._get_from_cache(article_id)
        if cached:
            logger.debug(f"快取命中: {article_id}")
            return cached

        # Step 2: 呼叫 LLM
        result = self._call_llm(title, content)
        if result is None:
            return None

        # Step 3: 存入快取
        self._save_to_cache(article_id, result)
        return result

    def batch_analyze(self, articles: list[dict], source: str = "ptt") -> list[dict]:
        """
        批次分析：一次送多篇給 LLM，大幅減少 API 呼叫次數。

        :param articles: 文章 list（需含 article_id, title 欄位）
        :param source: "ptt"（使用含鄉民俚語的 prompt）或 "news"（使用財經新聞 prompt）
        """
        import time

        results = []

        # 先處理快取命中的
        uncached = []
        for article in articles:
            article_id = article.get("article_id", "")
            cached = self._get_from_cache(article_id)
            if cached:
                enriched = article.copy()
                enriched["sentiment_score"] = cached["score"]
                enriched["sentiment_label"] = cached["label"]
                enriched["sentiment_reasoning"] = cached["reasoning"]
                enriched["analyzed_at"] = "cached"
                results.append(enriched)
            else:
                uncached.append(article)

        logger.info(f"快取命中 {len(results)} 篇，需分析 {len(uncached)} 篇")

        # 每批 10 篇一起送
        batch_size = 10
        for i in range(0, len(uncached), batch_size):
            batch = uncached[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(uncached) + batch_size - 1) // batch_size
            logger.info(
                f"分析批次 {batch_num}/{total_batches}（每批 {len(batch)} 篇）...")

            batch_results = self._call_llm_batch(batch, source=source)

            for article, sentiment in zip(batch, batch_results):
                enriched = article.copy()
                if sentiment:
                    enriched["sentiment_score"] = sentiment["score"]
                    enriched["sentiment_label"] = sentiment["label"]
                    enriched["sentiment_reasoning"] = sentiment["reasoning"]
                    enriched["analyzed_at"] = datetime.now().isoformat()
                    self._save_to_cache(article.get(
                        "article_id", ""), sentiment)
                else:
                    enriched["sentiment_score"] = None
                    enriched["sentiment_label"] = "error"
                    enriched["sentiment_reasoning"] = "分析失敗"
                results.append(enriched)

            time.sleep(1)

        logger.info(f"批次分析完成: {len(results)} 篇")
        return results

    def save_results(self, articles: list[dict], filename: str = "ptt_with_sentiment.jsonl"):
        """儲存情緒標註結果"""
        filepath = PROCESSED_DIR / filename
        with open(filepath, "w", encoding="utf-8") as f:
            for article in articles:
                article_copy = article.copy()
                if isinstance(article_copy.get("datetime"), datetime):
                    article_copy["datetime"] = article_copy["datetime"].isoformat()
                f.write(json.dumps(article_copy, ensure_ascii=False) + "\n")
        logger.info(f"已儲存 {len(articles)} 篇情緒標註結果至 {filepath}")

    def get_cache_stats(self) -> dict:
        """查看快取統計"""
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM sentiment_cache")
        count = cursor.fetchone()[0]
        conn.close()
        return {"cached_articles": count, "db_path": str(self.cache_db_path)}

    # ================================================================== #
    #  LLM 呼叫
    # ================================================================== #

    def _call_llm(self, title: str, content: str = "") -> Optional[dict]:
        """呼叫 Gemini API，含重試機制"""
        if not GEMINI_API_KEY:
            logger.warning("未設定 GEMINI_API_KEY，使用模擬模式")
            return self._mock_sentiment(title)

        import time

        for attempt in range(3):
            try:
                if self.model is None:
                    import google.generativeai as genai
                    genai.configure(api_key=GEMINI_API_KEY)
                    self.model = genai.GenerativeModel("gemini-2.5-flash")

                user_message = f"文章標題：{title}"
                if content:
                    truncated = content[:500] if len(
                        content) > 500 else content
                    user_message += f"\n\n文章內文（節錄）：\n{truncated}"

                response = self.model.generate_content(
                    f"{SENTIMENT_PROMPT}\n\n---\n\n{user_message}"
                )

                raw_text = response.text.strip()
                return self._parse_llm_response(raw_text)

            except Exception as e:
                wait = 10 * (attempt + 1)  # 10秒、20秒、30秒
                logger.warning(f"API 錯誤 (第{attempt+1}次): {e}, {wait}秒後重試...")
                time.sleep(wait)

        logger.error(f"3 次重試都失敗: {title[:30]}")
        return None

    def _call_llm_batch(self, articles: list[dict], source: str = "ptt") -> list[Optional[dict]]:
        """一次送多篇文章給 LLM 分析"""
        if not GEMINI_API_KEY:
            return [self._mock_sentiment(a.get("title", "")) for a in articles]

        import time

        # 根據來源選擇 prompt
        base_prompt = SENTIMENT_PROMPT if source == "ptt" else NEWS_SENTIMENT_PROMPT

        # 組合多篇文章成一個 prompt
        lines = []
        for idx, article in enumerate(articles):
            title = article.get("title", "")
            lines.append(f"文章{idx + 1}：{title}")

        batch_prompt = (
            f"{base_prompt}\n\n"
            f"---\n\n"
            f"請分析以下 {len(articles)} 篇文章，每篇各回傳一個 JSON 物件。\n"
            f"回傳格式：一個 JSON 陣列，包含 {len(articles)} 個物件。\n"
            f"例如：[{{\"score\": 0.5, \"label\": \"bullish\", \"reasoning\": \"...\"}}, ...]\n\n"
            + "\n".join(lines)
        )

        for attempt in range(3):
            try:
                if self.model is None:
                    import google.generativeai as genai
                    genai.configure(api_key=GEMINI_API_KEY)
                    self.model = genai.GenerativeModel("gemini-2.5-flash")

                response = self.model.generate_content(batch_prompt)
                raw_text = response.text.strip()

                # 解析 JSON 陣列
                cleaned = raw_text.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split(
                        "\n", 1)[1] if "\n" in cleaned else cleaned
                    cleaned = cleaned.rsplit("```", 1)[0]
                cleaned = cleaned.strip()

                import json
                data_list = json.loads(cleaned)

                if not isinstance(data_list, list):
                    data_list = [data_list]

                # 驗證每個結果
                results = []
                for data in data_list:
                    score = float(data.get("score", 0))
                    score = max(-1.0, min(1.0, score))
                    label = data.get("label", "neutral")
                    valid_labels = {"very_bearish", "bearish",
                                    "neutral", "bullish", "very_bullish"}
                    if label not in valid_labels:
                        label = "neutral"
                    results.append({
                        "score": round(score, 2),
                        "label": label,
                        "reasoning": data.get("reasoning", ""),
                    })

                # 補齊數量（如果 LLM 回傳不夠）
                while len(results) < len(articles):
                    results.append(None)

                return results[:len(articles)]

            except Exception as e:
                wait = 15 * (attempt + 1)
                logger.warning(
                    f"批次 API 錯誤 (第{attempt+1}次): {e}, {wait}秒後重試...")
                time.sleep(wait)

        return [None] * len(articles)

    @staticmethod
    def _parse_llm_response(raw_text: str) -> Optional[dict]:
        """解析 LLM 回傳的 JSON"""
        try:
            # 清理 markdown 標記
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split(
                    "\n", 1)[1] if "\n" in cleaned else cleaned
                cleaned = cleaned.rsplit("```", 1)[0]
            cleaned = cleaned.strip()

            data = json.loads(cleaned)

            # 驗證並修正
            score = float(data.get("score", 0))
            score = max(-1.0, min(1.0, score))  # 限制範圍

            label = data.get("label", "neutral")
            valid_labels = {"very_bearish", "bearish",
                            "neutral", "bullish", "very_bullish"}
            if label not in valid_labels:
                label = "neutral"

            reasoning = data.get("reasoning", "")

            return {
                "score": round(score, 2),
                "label": label,
                "reasoning": reasoning,
            }
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"LLM 回應解析失敗: {e}, 原始回應: {raw_text[:200]}")
            return None

    @staticmethod
    def _mock_sentiment(title: str) -> dict:
        """模擬模式：沒有 API Key 時用簡單規則估算"""
        bullish_words = ["多", "漲", "買", "噴", "利多", "看好", "突破", "創高", "外資買"]
        bearish_words = ["空", "跌", "賣", "丸", "利空", "看壞", "崩", "破", "外資賣"]

        score = 0.0
        for w in bullish_words:
            if w in title:
                score += 0.2
        for w in bearish_words:
            if w in title:
                score -= 0.2
        score = max(-1.0, min(1.0, score))

        if score > 0.6:
            label = "very_bullish"
        elif score > 0.2:
            label = "bullish"
        elif score > -0.2:
            label = "neutral"
        elif score > -0.6:
            label = "bearish"
        else:
            label = "very_bearish"

        return {"score": round(score, 2), "label": label, "reasoning": "[模擬模式] 基於關鍵字規則判斷"}

    # ================================================================== #
    #  SQLite 快取
    # ================================================================== #

    def _init_cache_db(self):
        """初始化快取資料庫"""
        conn = sqlite3.connect(self.cache_db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_cache (
                article_id TEXT PRIMARY KEY,
                score REAL,
                label TEXT,
                reasoning TEXT,
                created_at TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _get_from_cache(self, article_id: str) -> Optional[dict]:
        """從快取查詢"""
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.execute(
            "SELECT score, label, reasoning FROM sentiment_cache WHERE article_id = ?",
            (article_id,),
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            return {"score": row[0], "label": row[1], "reasoning": row[2]}
        return None

    def _save_to_cache(self, article_id: str, result: dict):
        """存入快取"""
        conn = sqlite3.connect(self.cache_db_path)
        conn.execute(
            """INSERT OR REPLACE INTO sentiment_cache
               (article_id, score, label, reasoning, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (article_id, result["score"], result["label"],
             result["reasoning"], datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()


# ================================================================== #
#  測試區塊
# ================================================================== #
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()

    test_titles = [
        "[標的] 2330 台積電 多",
        "[閒聊] 台積電丸子 咕嚕咕嚕了",
        "[新聞] 台積電法說會 毛利率創新高",
        "[標的] 2330 GG 外資提款機 空",
        "[心得] 護國神山要噴了 春燕來了",
    ]

    print("=" * 60)
    print("LLM 情緒分析測試")
    print("=" * 60)

    for i, title in enumerate(test_titles):
        result = analyzer.analyze(f"test_{i}", title)
        if result:
            if result["score"] > 0.2:
                icon = "🟢"
            elif result["score"] < -0.2:
                icon = "🔴"
            else:
                icon = "⚪"
            print(f"\n{icon} {title}")
            print(f"   分數: {result['score']:+.2f} | 標籤: {result['label']}")
            print(f"   理由: {result['reasoning']}")

    print(f"\n--- 快取統計 ---")
    stats = analyzer.get_cache_stats()
    print(f"已快取: {stats['cached_articles']} 篇")

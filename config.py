import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
# === 路徑設定 ===
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FINAL_DIR = DATA_DIR / "final"
# 確保目錄存在
for d in [RAW_DIR, PROCESSED_DIR, FINAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)
# === 股票設定 ===
STOCK_TICKER = "2330.TW"          # yfinance 格式
STOCK_CODE = "2330"               # TWSE API 格式
STOCK_NAME = "台積電"

# === PTT 爬蟲設定 ===
PTT_BOARD = "Stock"
PTT_BASE_URL = "https://www.ptt.cc"
PTT_SEARCH_URL = f"{PTT_BASE_URL}/bbs/{PTT_BOARD}/search"
PTT_COOKIES = {"over18": "1"}
PTT_REQUEST_DELAY = 1.5           # 秒，爬蟲禮儀
PTT_MAX_RETRIES = 3

# 台積電相關關鍵字（同義詞表）
TSMC_KEYWORDS = [
    "台積電", "台積", "2330", "GG", "積積",
    "護國神山", "神山", "TSM", "TSMC",
]

# PTT 文章分類標籤
PTT_CATEGORIES = ["標的", "新聞", "閒聊", "心得", "請益", "情報", "其他"]

# === LLM 設定（Stage 2 才會用到）===
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = "claude-sonnet-4-20250514"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# === 鉅亨網 (Anue) 設定 ===
CNYES_API_URL = "https://api.cnyes.com/media/api/v1/newslist/category/tw_stock"
CNYES_PAGE_SIZE = 30
CNYES_REQUEST_DELAY = 1.0

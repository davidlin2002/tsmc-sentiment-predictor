# 台股情緒分析與趨勢預測 —

## 一、專案總覽

**目標：** 探索 PTT 股版輿情與台積電（2330.TW）股價波動之間的關聯性

**定位：** 這不是一個「預測股價」的專案，而是一個展示「非結構化資料處理 → 特徵工程 → 機器學習 pipeline」完整工程能力的作品集專案。

---

## 二、專案目錄結構

```
tsmc-sentiment-predictor/
│
├── README.md                    # 專案說明（面試官第一眼看的）
├── requirements.txt             # 依賴套件
├── config.py                    # 全域設定（API key、路徑、參數）
├── .env                         # 環境變數（API keys，不進 git）
├── .gitignore
│
├── data/                        # 所有資料（不進 git，README 說明如何重建）
│   ├── raw/                     # 原始資料，不修改
│   │   ├── ptt_articles.jsonl   # PTT 爬蟲原始結果
│   │   └── stock_price.csv      # 股價原始資料
│   ├── processed/               # 清洗後的資料
│   │   ├── ptt_with_sentiment.jsonl   # 加上情緒分數的文章
│   │   ├── daily_sentiment.csv        # 每日情緒聚合
│   │   └── stock_clean.csv            # 清洗後股價
│   └── final/                   # 模型輸入用的最終資料
│       └── features.csv         # 合併後的特徵表
│
├── src/                         # 核心程式碼
│   ├── __init__.py
│   ├── scraper/
│   │   ├── __init__.py
│   │   ├── ptt_scraper.py       # PTT 爬蟲（進入文章抓完整時間戳）
│   │   └── stock_fetcher.py     # 股價抓取（yfinance + 備案）
│   ├── sentiment/
│   │   ├── __init__.py
│   │   ├── llm_analyzer.py      # LLM 情緒分析（含快取機制）
│   │   └── prompt_templates.py  # Prompt 模板管理
│   ├── features/
│   │   ├── __init__.py
│   │   ├── time_aligner.py      # 時間對齊邏輯（核心難點）
│   │   └── feature_engineer.py  # 特徵工程
│   └── model/
│       ├── __init__.py
│       ├── trainer.py           # 模型訓練（Time Series Split）
│       └── evaluator.py         # 模型評估與解釋
│
├── notebooks/                   # 探索與展示用
│   ├── 01_eda.ipynb             # 探索性分析
│   ├── 02_sentiment_demo.ipynb  # LLM 情緒分析 demo
│   └── 03_model_results.ipynb   # 模型結果視覺化
│
├── app/                         # Streamlit 展示
│   └── streamlit_app.py
│
└── tests/                       # 測試（加分項）
    ├── test_scraper.py
    └── test_features.py
```

---

## 三、資料流設計（Data Contract）

這是 Gemini 完全沒做的部分。每一階段的輸入/輸出格式必須在寫 code 之前定義清楚，否則後面一定會返工。

### Stage 1 → PTT 原始資料 (`ptt_articles.jsonl`)

每一行是一篇文章，格式：

```json
{
  "article_id": "M.1711234567.A.B12",
  "title": "[標的] 2330 台積電 多",
  "author": "stock_king",
  "datetime": "2025-03-27T14:23:00+08:00",
  "content": "如題，法人連買三天，外資回補...",
  "push_count": {"推": 15, "噓": 3, "→": 8},
  "url": "https://www.ptt.cc/bbs/Stock/M.1711234567.A.B12.html",
  "category": "標的"
}
```

**關鍵設計決策：**

- 使用 JSONL（每行一筆 JSON）而非 CSV，因為文章內容含有逗號、換行符
- `datetime` 使用 ISO 8601 含時區，從文章內頁的 meta tag 抓取，解決 Gemini 只有「月/日」的問題
- `article_id` 從 URL 提取，用來做去重和 LLM 快取的 key
- `push_count` 分開記錄推/噓/箭頭，這本身就是很好的情緒特徵
- `category` 從標題的 `[標的]`、`[新聞]`、`[閒聊]` 等 tag 解析

### Stage 1 → 股價資料 (`stock_price.csv`)

```
date,open,high,low,close,volume,change_pct
2025-03-24,890.0,905.0,888.0,903.0,45231000,1.57
2025-03-25,905.0,910.0,895.0,898.0,38120000,-0.55
```

**關鍵設計決策：**

- `date` 是交易日（只有開盤日）
- `change_pct` 在這裡就算好，後面直接用
- 欄位全小寫 + snake_case，跟 Python 慣例一致

### Stage 2 → 情緒標註結果 (`ptt_with_sentiment.jsonl`)

在原始文章資料上，新增：

```json
{
  "article_id": "M.1711234567.A.B12",
  "title": "[標的] 2330 台積電 多",
  "datetime": "2025-03-27T14:23:00+08:00",
  "sentiment_score": 0.72,
  "sentiment_label": "bullish",
  "sentiment_reasoning": "作者看多台積電，提及法人買超和外資回補，屬利多訊號",
  "llm_model": "claude-sonnet-4-20250514",
  "analyzed_at": "2025-03-27T16:00:00+08:00"
}
```

**關鍵設計決策：**

- `sentiment_score`：-1.0 到 1.0，連續值
- `sentiment_label`：分類標籤 (very_bearish / bearish / neutral / bullish / very_bullish)
- `sentiment_reasoning`：LLM 的判斷理由（debug 用，也可以寫進 README 展示）
- `llm_model` + `analyzed_at`：可重現性（Reproducibility），面試加分
- 保留 `article_id` 做為 LLM 快取的 key，避免重複呼叫浪費錢

### Stage 3 → 每日情緒聚合 (`daily_sentiment.csv`)

這是「時間對齊」後的結果，也是整個專案最關鍵的中間產物：

```
trade_date,article_count,avg_sentiment,max_sentiment,min_sentiment,sentiment_std,bullish_ratio,push_net_avg
2025-03-24,12,0.35,0.9,-0.3,0.28,0.67,8.5
2025-03-25,8,0.12,0.6,-0.7,0.41,0.50,3.2
```

**時間對齊規則（核心邏輯）：**

```
                    PTT 文章時間               對應交易日
                    ─────────────              ──────────
週一 ~ 週四          當天 00:00 ~ 23:59    →    隔天開盤日
週五                 00:00 ~ 23:59         →    下週一
週六                 00:00 ~ 23:59         →    下週一
週日                 00:00 ~ 23:59         →    下週一

原因：今天的輿論影響的是「明天」的股價，不是今天的。
週末累積的情緒會在週一一次爆發。
```

這個對齊規則是整個專案最值得在面試時討論的設計決策。

### Stage 4 → 最終特徵表 (`features.csv`)

```
trade_date,close,change_pct,avg_sentiment,sentiment_std,bullish_ratio,
article_count,push_net_avg,sentiment_lag1,sentiment_lag2,sentiment_lag3,
sma_5,sma_20,ema_12,volume_change_pct,sentiment_momentum,
label
2025-03-25,898.0,-0.55,0.35,0.28,0.67,12,8.5,0.20,0.45,0.10,
895.0,880.5,890.2,15.3,0.15,
0
```

**特徵分為三大類：**

1. **情緒特徵（Sentiment Features）**
    - avg_sentiment：當日平均情緒
    - sentiment_std：情緒分歧度（越大代表多空分歧越嚴重）
    - bullish_ratio：看多文章佔比
    - article_count：討論熱度（文章越多可能代表即將有大波動）
    - push_net_avg：平均淨推文數（推 - 噓）
    - sentiment_lag1/2/3：前 1/2/3 天的情緒（滯後效應）
    - sentiment_momentum：情緒動能（近 3 天 vs 近 7 天平均）
2. **技術指標（Technical Features）**
    - sma_5, sma_20：5 日 / 20 日簡單移動平均
    - ema_12：12 日指數移動平均
    - volume_change_pct：成交量變化率
3. **標籤（Label）**
    - 二元分類：隔天漲 = 1，跌 = 0
    - 門檻：漲跌幅 > ±0.5% 才算明確漲跌，中間的當作 noise 移除
    （這個設計決策可以面試時討論，展現你對問題的思考深度）

---

## 四、Gemini 方案的問題與我的改進

| 問題 | Gemini 的做法 | 改進方案 |
| --- | --- | --- |
| PTT 日期只有月/日 | 直接用列表頁的日期 | 進入文章內頁抓 `<meta>` 的完整 timestamp |
| 關鍵字過濾 | 硬寫 `"台積" or "2330"` | 建立同義詞表：台積、2330、GG、積積、護國神山、TSM |
| 時間對齊 | 完全沒處理 | 明確定義 T+1 對齊規則，週末歸入週一 |
| LLM 成本控制 | 只在 README 提 | 實作 SQLite 快取，article_id 為 key，呼叫前先查 |
| yfinance 不穩定 | 沒有備案 | 主要用 yfinance，備案用 TWSE OpenAPI |
| 資料格式 | 沒有定義 | 完整 Data Contract，每階段輸入輸出格式明確 |
| 推噓文 | 完全忽略 | 推/噓/箭頭分開計算，本身就是情緒特徵 |
| 文章分類 | 沒處理 | 解析 [標的]、[新聞]、[閒聊] 等分類 tag |

---

## 五、開發順序建議

```
Week 1：先跑通最小可行 pipeline（各階段各 10 筆資料）
  ├── Day 1-2：PTT 爬蟲 + 股價抓取
  ├── Day 3-4：LLM 情緒分析 + 快取機制
  └── Day 5：時間對齊 + 手動驗證結果合理性

Week 2：擴大資料量 + 特徵工程 + 建模
  ├── Day 1-2：爬 6 個月資料，跑完 LLM 標註
  ├── Day 3-4：特徵工程 + XGBoost 訓練
  └── Day 5：模型評估 + 特徵重要性分析

Week 3：展示層 + 文件
  ├── Day 1-2：Streamlit app
  ├── Day 3：EDA notebook 整理
  └── Day 4-5：README + GitHub 排版
```

---

## 六、面試可討論的設計決策（準備好回答）

1. **為什麼用 T+1 對齊？** 因為我們要「預測」，不能用當天資訊預測當天。
2. **為什麼用分類不用回歸？** 股價的絕對數值很難預測，但方向性判斷更有實用價值。
3. **為什麼要設 ±0.5% 的門檻？** 減少 noise，讓模型學到的是有意義的趨勢變化。
4. **LLM vs TextBlob？** PTT 鄉民語言太特殊，傳統 NLP 工具完全無法處理。
5. **為什麼用 Time Series Split？** 時間序列不能用隨機 CV，否則等於用未來預測過去。
6. **模型準確率不高怎麼辦？** 這個專案的價值在 pipeline，不在預測準確率。如果情緒特徵的 feature importance 排名靠前，就已經證明了假設。

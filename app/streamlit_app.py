# isort: skip_file
"""
Streamlit 互動展示介面
"""

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import pandas as pd
import json
from config import STOCK_NAME, RAW_DIR, PROCESSED_DIR, FINAL_DIR
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


# === 頁面設定 ===
st.set_page_config(
    page_title=f"{STOCK_NAME} PTT 輿情分析",
    page_icon="🏔️",
    layout="wide",
)

st.title(f"🏔️ {STOCK_NAME} PTT 輿情分析與趨勢預測")
st.caption("結合 PTT 鄉民情緒 × 鉅亨財經新聞 × 技術指標，預測台積電隔日漲跌")


# ================================================================== #
#  資料載入
# ================================================================== #
@st.cache_data
def load_data():
    data = {}

    price_path = RAW_DIR / "stock_price.csv"
    if price_path.exists():
        data["price"] = pd.read_csv(price_path, index_col=0, parse_dates=True)

    sentiment_path = PROCESSED_DIR / "daily_sentiment.csv"
    if sentiment_path.exists():
        data["sentiment"] = pd.read_csv(
            sentiment_path, index_col=0, parse_dates=True)

    news_sentiment_path = PROCESSED_DIR / "daily_news_sentiment.csv"
    if news_sentiment_path.exists():
        data["news_sentiment"] = pd.read_csv(
            news_sentiment_path, index_col=0, parse_dates=True)

    features_path = FINAL_DIR / "features.csv"
    if features_path.exists():
        data["features"] = pd.read_csv(
            features_path, index_col=0, parse_dates=True)

    results_path = FINAL_DIR / "model_results.json"
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            data["model_results"] = json.load(f)

    ablation_path = FINAL_DIR / "ablation_results.json"
    if ablation_path.exists():
        with open(ablation_path, "r", encoding="utf-8") as f:
            data["ablation_results"] = json.load(f)

    articles_path = PROCESSED_DIR / "ptt_with_sentiment.jsonl"
    if articles_path.exists():
        articles = []
        with open(articles_path, "r", encoding="utf-8") as f:
            for line in f:
                articles.append(json.loads(line))
        data["articles"] = articles

    cnyes_path = PROCESSED_DIR / "cnyes_with_sentiment.jsonl"
    if cnyes_path.exists():
        cnyes_articles = []
        with open(cnyes_path, "r", encoding="utf-8") as f:
            for line in f:
                cnyes_articles.append(json.loads(line))
        data["cnyes_articles"] = cnyes_articles

    return data


data = load_data()

if not data:
    st.error("找不到資料檔案！請先執行 `python main.py` 產生資料。")
    st.stop()


# ================================================================== #
#  Tab 分頁
# ================================================================== #
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 股價與情緒", "📊 模型結果", "🧪 Ablation Study", "📝 文章瀏覽", "🔍 即時分析"
])


# ================================================================== #
#  Tab 1: 股價 + 情緒走勢
# ================================================================== #
with tab1:
    if "price" not in data:
        st.warning("找不到股價資料")
    else:
        price_df = data["price"]
        sent_df = data.get("sentiment")
        news_df = data.get("news_sentiment")

        # --- 日期範圍選擇 ---
        min_date = price_df.index.min().date()
        max_date = price_df.index.max().date()
        col_l, col_r = st.columns([1, 3])
        with col_l:
            date_range = st.select_slider(
                "顯示期間",
                options=["近 1 個月", "近 3 個月", "近 6 個月", "全部"],
                value="近 3 個月",
            )

        cutoff = {
            "近 1 個月": pd.Timestamp.now() - pd.DateOffset(months=1),
            "近 3 個月": pd.Timestamp.now() - pd.DateOffset(months=3),
            "近 6 個月": pd.Timestamp.now() - pd.DateOffset(months=6),
            "全部": pd.Timestamp("2000-01-01"),
        }[date_range]

        price_filtered = price_df[price_df.index >= cutoff]
        sent_filtered = sent_df[sent_df.index >=
                                cutoff] if sent_df is not None else None
        news_filtered = news_df[news_df.index >=
                                cutoff] if news_df is not None else None

        # --- 主圖：K線 + 成交量 + 情緒 ---
        has_volume = "volume" in price_filtered.columns
        row_heights = [0.55, 0.15, 0.3] if has_volume else [0.65, 0.35]
        row_count = 3 if has_volume else 2
        subplot_titles = [f"{STOCK_NAME} 股價走勢"]
        if has_volume:
            subplot_titles.append("成交量")
        subplot_titles.append("每日情緒分數")

        fig = make_subplots(
            rows=row_count, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=row_heights,
            subplot_titles=subplot_titles,
        )

        # K 線
        fig.add_trace(go.Candlestick(
            x=price_filtered.index,
            open=price_filtered["open"],
            high=price_filtered["high"],
            low=price_filtered["low"],
            close=price_filtered["close"],
            name="K線",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ), row=1, col=1)

        # 成交量
        if has_volume:
            colors = [
                "#26a69a" if c >= o else "#ef5350"
                for c, o in zip(price_filtered["close"], price_filtered["open"])
            ]
            fig.add_trace(go.Bar(
                x=price_filtered.index,
                y=price_filtered["volume"],
                name="成交量",
                marker_color=colors,
                showlegend=False,
            ), row=2, col=1)

        sentiment_row = 3 if has_volume else 2

        # PTT 情緒
        if sent_filtered is not None and not sent_filtered.empty:
            fig.add_trace(go.Scatter(
                x=sent_filtered.index,
                y=sent_filtered["avg_sentiment"],
                mode="lines+markers",
                name="PTT 情緒",
                line=dict(color="orange", width=2),
                marker=dict(size=5),
            ), row=sentiment_row, col=1)

            # 極端值標記
            bull_pts = sent_filtered[sent_filtered["avg_sentiment"] > 0.5]
            bear_pts = sent_filtered[sent_filtered["avg_sentiment"] < -0.5]
            if not bull_pts.empty:
                fig.add_trace(go.Scatter(
                    x=bull_pts.index, y=bull_pts["avg_sentiment"],
                    mode="markers", name="極樂觀",
                    marker=dict(color="#26a69a", size=14,
                                symbol="triangle-up"),
                ), row=sentiment_row, col=1)
            if not bear_pts.empty:
                fig.add_trace(go.Scatter(
                    x=bear_pts.index, y=bear_pts["avg_sentiment"],
                    mode="markers", name="極悲觀",
                    marker=dict(color="#ef5350", size=14,
                                symbol="triangle-down"),
                ), row=sentiment_row, col=1)

        # 鉅亨情緒
        if news_filtered is not None and not news_filtered.empty:
            fig.add_trace(go.Scatter(
                x=news_filtered.index,
                y=news_filtered["avg_sentiment"],
                mode="lines+markers",
                name="鉅亨新聞情緒",
                line=dict(color="royalblue", width=2, dash="dot"),
                marker=dict(size=4),
            ), row=sentiment_row, col=1)

        fig.add_hline(y=0, line_dash="dash", line_color="gray",
                      row=sentiment_row, col=1)

        fig.update_layout(
            height=680,
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom",
                        y=1.01, xanchor="right", x=1),
            margin=dict(t=60),
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- 統計摘要列 ---
        latest = price_filtered.iloc[-1]
        earliest = price_filtered.iloc[0]
        period_return = (latest["close"] -
                         earliest["close"]) / earliest["close"] * 100

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("最新收盤", f"${latest['close']:,.0f}",
                  f"{latest.get('change_pct', 0):+.2f}%" if "change_pct" in latest else None)
        c2.metric("期間報酬", f"{period_return:+.1f}%")
        c3.metric("最高價", f"${price_filtered['high'].max():,.0f}")
        c4.metric("最低價", f"${price_filtered['low'].min():,.0f}")

        if sent_filtered is not None and not sent_filtered.empty:
            avg_sent = sent_filtered["avg_sentiment"].mean()
            c5.metric("PTT 平均情緒", f"{avg_sent:+.3f}",
                      "偏多" if avg_sent > 0 else "偏空")

        # --- 情緒與報酬相關分析 ---
        if sent_filtered is not None and "features" in data:
            st.divider()
            feat_df = data["features"]
            feat_filtered = feat_df[feat_df.index >= cutoff]

            if not feat_filtered.empty and "avg_sentiment" in feat_filtered.columns and "label" in feat_filtered.columns:
                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown("**情緒分數 vs 隔日漲跌**")
                    scatter_df = feat_filtered[[
                        "avg_sentiment", "label", "next_change_pct"]].dropna()
                    scatter_df["漲跌"] = scatter_df["label"].map(
                        {1.0: "漲", 0.0: "跌"})
                    fig_sc = px.scatter(
                        scatter_df,
                        x="avg_sentiment", y="next_change_pct",
                        color="漲跌",
                        color_discrete_map={"漲": "#26a69a", "跌": "#ef5350"},
                        labels={"avg_sentiment": "PTT 平均情緒",
                                "next_change_pct": "隔日漲跌 (%)"},
                        height=320,
                    )
                    fig_sc.add_vline(x=0, line_dash="dash", line_color="gray")
                    fig_sc.add_hline(y=0, line_dash="dash", line_color="gray")
                    st.plotly_chart(fig_sc, use_container_width=True)

                with col_b:
                    st.markdown("**每日文章數量**")
                    if "article_count" in feat_filtered.columns:
                        fig_bar = px.bar(
                            feat_filtered.reset_index(),
                            x="trade_date", y="article_count",
                            labels={"trade_date": "日期",
                                    "article_count": "PTT 文章數"},
                            color_discrete_sequence=["#f4a621"],
                            height=320,
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)


# ================================================================== #
#  Tab 2: 模型結果
# ================================================================== #
with tab2:
    if "model_results" not in data:
        st.warning("找不到模型結果，請先執行 `python main.py`")
    else:
        results = data["model_results"]

        # --- KPI ---
        st.subheader("XGBoost 模型評估（Time Series Split）")
        c1, c2, c3, c4 = st.columns(4)
        avg_acc = results["avg_accuracy"]
        avg_f1 = results["avg_f1"]

        c1.metric("平均準確率", f"{avg_acc:.2%}",
                  f"{(avg_acc - 0.5) * 100:+.1f}pp vs 隨機基準")
        c2.metric("平均 F1 Score", f"{avg_f1:.2%}")
        c3.metric("訓練樣本", f"{results['n_samples']} 筆")
        c4.metric("特徵數量", f"{results['n_features']} 個")

        st.caption("基準線為隨機猜測（50%）。股價預測困難，能穩定高於基準線即有意義。")

        # --- 折線圖：各折準確率 ---
        fold_df = pd.DataFrame(results["fold_results"])
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("**各折 Accuracy vs F1**")
            fig_fold = go.Figure()
            fig_fold.add_trace(go.Scatter(
                x=fold_df["fold"], y=fold_df["accuracy"],
                mode="lines+markers", name="Accuracy",
                line=dict(color="steelblue", width=2),
                marker=dict(size=8),
            ))
            fig_fold.add_trace(go.Scatter(
                x=fold_df["fold"], y=fold_df["f1"],
                mode="lines+markers", name="F1",
                line=dict(color="orange", width=2),
                marker=dict(size=8),
            ))
            fig_fold.add_hline(y=0.5, line_dash="dash", line_color="red",
                               annotation_text="隨機基準 50%")
            fig_fold.update_layout(
                height=320,
                xaxis_title="Fold",
                yaxis_title="Score",
                yaxis=dict(range=[0.2, 0.9]),
                legend=dict(orientation="h"),
                margin=dict(t=20),
            )
            st.plotly_chart(fig_fold, use_container_width=True)

        with col_right:
            st.markdown("**各折 Precision vs Recall**")
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Bar(
                x=fold_df["fold"], y=fold_df["precision"],
                name="Precision", marker_color="steelblue", opacity=0.8,
            ))
            fig_pr.add_trace(go.Bar(
                x=fold_df["fold"], y=fold_df["recall"],
                name="Recall", marker_color="coral", opacity=0.8,
            ))
            fig_pr.update_layout(
                height=320,
                barmode="group",
                xaxis_title="Fold",
                yaxis_title="Score",
                yaxis=dict(range=[0, 1]),
                legend=dict(orientation="h"),
                margin=dict(t=20),
            )
            st.plotly_chart(fig_pr, use_container_width=True)

        # --- 各折詳細數據表 ---
        with st.expander("查看各折詳細數據"):
            display_fold = fold_df.copy()
            display_fold.columns = ["Fold", "訓練筆數", "驗證筆數",
                                    "Accuracy", "F1", "Precision", "Recall"]
            st.dataframe(
                display_fold.style.format({
                    "Accuracy": "{:.2%}", "F1": "{:.2%}",
                    "Precision": "{:.2%}", "Recall": "{:.2%}",
                }),
                use_container_width=True,
                hide_index=True,
            )

        # --- 特徵重要性 ---
        st.divider()
        st.subheader("特徵重要性")
        st.caption("排名越前代表模型越依賴該特徵。情緒特徵若排名靠前，代表 PTT 輿情確實帶有預測資訊。")

        if "feature_importance" in results:
            fi_df = pd.DataFrame(results["feature_importance"])

            # 標記特徵類型
            def classify_feature(name):
                if name.startswith("news_") or name == "sentiment_divergence":
                    return "鉅亨新聞情緒"
                elif any(k in name for k in ["sentiment", "bullish", "push", "article", "discussion"]):
                    return "PTT 情緒"
                elif any(k in name for k in ["sma", "ema", "volume", "volatility", "intraday", "price_vs"]):
                    return "技術指標"
                else:
                    return "其他"

            fi_df["類型"] = fi_df["feature"].apply(classify_feature)
            fi_df = fi_df.sort_values("importance", ascending=True).tail(20)

            color_map = {
                "PTT 情緒": "#f4a621",
                "鉅亨新聞情緒": "royalblue",
                "技術指標": "#26a69a",
                "其他": "gray",
            }

            fig_fi = px.bar(
                fi_df,
                x="importance", y="feature",
                color="類型",
                color_discrete_map=color_map,
                orientation="h",
                labels={"importance": "重要性分數", "feature": "特徵名稱"},
                height=520,
            )
            fig_fi.update_layout(
                margin=dict(l=180, t=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.01),
            )
            st.plotly_chart(fig_fi, use_container_width=True)

            # 各類型重要性佔比
            st.markdown("**各類特徵重要性佔比**")
            type_imp = (
                pd.DataFrame(results["feature_importance"])
                .assign(類型=lambda df: df["feature"].apply(classify_feature))
                .groupby("類型")["importance"]
                .sum()
                .reset_index()
                .rename(columns={"importance": "合計重要性"})
                .sort_values("合計重要性", ascending=False)
            )
            fig_pie = px.pie(
                type_imp,
                values="合計重要性",
                names="類型",
                color="類型",
                color_discrete_map=color_map,
                height=280,
            )
            fig_pie.update_traces(textposition="inside",
                                  textinfo="percent+label")
            fig_pie.update_layout(showlegend=False, margin=dict(t=20))
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("請重新執行 `python main.py --stage model` 以產生特徵重要性資料。")


# ================================================================== #
#  Tab 3: Ablation Study
# ================================================================== #
with tab3:
    if "ablation_results" not in data:
        st.warning("尚未執行 Ablation Study，請執行：")
        st.code("python main.py --stage ablation")
    else:
        ab = data["ablation_results"]

        st.subheader("Ablation Study — 情緒特徵貢獻度驗證")
        st.caption(
            "固定使用 XGBoost，逐步加入不同特徵組合，驗證「PTT 情緒是否帶來預測提升」。"
            "若 C > A，代表情緒特徵確實有貢獻。"
        )

        abl = ab.get("ablation", {})
        groups = list(abl.keys())
        accs = [abl[g]["avg_accuracy"] for g in groups]
        f1s  = [abl[g]["avg_f1"] for g in groups]
        nf   = [abl[g]["n_features"] for g in groups]

        # 顏色：C 組（技術+PTT）高亮
        bar_colors = ["#90caf9", "#ffcc80", "#26a69a", "#80cbc4"]

        ab_col1, ab_col2 = st.columns(2)

        with ab_col1:
            st.markdown("**各特徵組合 Accuracy**")
            fig_abl_acc = go.Figure(go.Bar(
                x=groups, y=accs,
                marker_color=bar_colors,
                text=[f"{v:.1%}" for v in accs],
                textposition="outside",
            ))
            fig_abl_acc.add_hline(y=0.5, line_dash="dash", line_color="red",
                                  annotation_text="隨機基準 50%",
                                  annotation_position="bottom right")
            fig_abl_acc.update_layout(
                height=380, yaxis=dict(range=[0.3, 0.65], tickformat=".0%"),
                margin=dict(t=20, b=60), xaxis_tickangle=-20,
            )
            st.plotly_chart(fig_abl_acc, use_container_width=True)

        with ab_col2:
            st.markdown("**各特徵組合 F1 Score**")
            fig_abl_f1 = go.Figure(go.Bar(
                x=groups, y=f1s,
                marker_color=bar_colors,
                text=[f"{v:.1%}" for v in f1s],
                textposition="outside",
            ))
            fig_abl_f1.add_hline(y=0.5, line_dash="dash", line_color="red",
                                 annotation_text="隨機基準 50%",
                                 annotation_position="bottom right")
            fig_abl_f1.update_layout(
                height=380, yaxis=dict(range=[0.3, 0.65], tickformat=".0%"),
                margin=dict(t=20, b=60), xaxis_tickangle=-20,
            )
            st.plotly_chart(fig_abl_f1, use_container_width=True)

        # 結論摘要
        if len(accs) >= 3:
            delta_acc = accs[2] - accs[0]
            delta_f1  = f1s[2]  - f1s[0]
            conclusion = "✅ 加入情緒特徵後準確率提升" if delta_acc > 0 else "⚠️ 情緒特徵未帶來準確率提升"
            st.info(
                f"{conclusion}  \n"
                f"技術指標（A）→ 技術+PTT情緒（C）：Accuracy **{delta_acc:+.1%}**，F1 **{delta_f1:+.1%}**"
            )

        st.divider()

        # 多模型比較
        st.subheader("多模型比較 — 為什麼選 XGBoost？")
        st.caption("固定特徵組合（技術 + PTT 情緒），比較不同演算法的表現。")

        mc = ab.get("model_comparison", {})
        models = list(mc.keys())
        mc_accs = [mc[m]["avg_accuracy"] for m in models]
        mc_f1s  = [mc[m]["avg_f1"] for m in models]

        model_colors = {
            "Logistic Regression": "#90caf9",
            "Random Forest": "#ffcc80",
            "XGBoost": "#26a69a",
            "LightGBM": "#ce93d8",
        }
        mc_colors = [model_colors.get(m, "gray") for m in models]

        mc_col1, mc_col2 = st.columns(2)
        with mc_col1:
            st.markdown("**各模型 Accuracy**")
            fig_mc_acc = go.Figure(go.Bar(
                x=models, y=mc_accs,
                marker_color=mc_colors,
                text=[f"{v:.1%}" for v in mc_accs],
                textposition="outside",
            ))
            fig_mc_acc.add_hline(y=0.5, line_dash="dash", line_color="red")
            fig_mc_acc.update_layout(
                height=320, yaxis=dict(range=[0.3, 0.65], tickformat=".0%"),
                margin=dict(t=20),
            )
            st.plotly_chart(fig_mc_acc, use_container_width=True)

        with mc_col2:
            st.markdown("**各模型 F1 Score**")
            fig_mc_f1 = go.Figure(go.Bar(
                x=models, y=mc_f1s,
                marker_color=mc_colors,
                text=[f"{v:.1%}" for v in mc_f1s],
                textposition="outside",
            ))
            fig_mc_f1.add_hline(y=0.5, line_dash="dash", line_color="red")
            fig_mc_f1.update_layout(
                height=320, yaxis=dict(range=[0.3, 0.65], tickformat=".0%"),
                margin=dict(t=20),
            )
            st.plotly_chart(fig_mc_f1, use_container_width=True)

        # 各模型詳細表格
        with st.expander("查看各模型詳細數據"):
            mc_rows = []
            for m in models:
                fold_df = pd.DataFrame(mc[m]["fold_results"])
                mc_rows.append({
                    "模型": m,
                    "Accuracy": mc[m]["avg_accuracy"],
                    "F1": mc[m]["avg_f1"],
                    "Acc 標準差": round(fold_df["accuracy"].std(), 4),
                    "F1 標準差": round(fold_df["f1"].std(), 4),
                })
            mc_df = pd.DataFrame(mc_rows)
            st.dataframe(
                mc_df.style.format({
                    "Accuracy": "{:.2%}", "F1": "{:.2%}",
                    "Acc 標準差": "{:.4f}", "F1 標準差": "{:.4f}",
                }).highlight_max(subset=["Accuracy", "F1"], color="#c8e6c9"),
                use_container_width=True,
                hide_index=True,
            )


# ================================================================== #
#  Tab 4: 文章瀏覽
# ================================================================== #
with tab4:
    ptt_articles = data.get("articles", [])
    cnyes_articles = data.get("cnyes_articles", [])
    has_ptt = len(ptt_articles) > 0
    has_cnyes = len(cnyes_articles) > 0

    if not has_ptt and not has_cnyes:
        st.warning("找不到文章資料")
    else:
        # --- 篩選列 ---
        fc1, fc2, fc3 = st.columns([2, 2, 2])

        with fc1:
            src_opts = []
            if has_ptt:
                src_opts.append(f"PTT ({len(ptt_articles)} 篇)")
            if has_cnyes:
                src_opts.append(f"鉅亨新聞 ({len(cnyes_articles)} 篇)")
            if has_ptt and has_cnyes:
                src_opts.append("全部")
            source_filter = st.radio("資料來源", src_opts, horizontal=True)

        with fc2:
            sent_filter = st.radio(
                "情緒篩選",
                ["全部", "利多 (>0.2)", "中性", "利空 (<-0.2)"],
                horizontal=True,
            )

        with fc3:
            sort_by = st.radio(
                "排序", ["時間（新→舊）", "情緒分數（高→低）", "情緒分數（低→高）"], horizontal=True)

        # --- 組合文章池 ---
        if "鉅亨" in source_filter:
            pool = cnyes_articles
        elif "PTT" in source_filter:
            pool = ptt_articles
        else:
            pool = ptt_articles + cnyes_articles

        # 情緒篩選
        def sent_ok(a):
            s = a.get("sentiment_score")
            if s is None:
                return sent_filter == "全部"
            if sent_filter == "利多 (>0.2)":
                return s > 0.2
            elif sent_filter == "利空 (<-0.2)":
                return s < -0.2
            elif sent_filter == "中性":
                return -0.2 <= s <= 0.2
            return True

        pool = [a for a in pool if sent_ok(a)]

        # 排序
        def sort_key(a):
            if sort_by == "時間（新→舊）":
                return a.get("datetime", "") or ""
            s = a.get("sentiment_score") or 0
            return s if "低→高" in sort_by else -s

        pool = sorted(pool, key=sort_key, reverse=("新→舊" in sort_by))

        # --- 分頁 ---
        PAGE_SIZE = 30
        total = len(pool)
        total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)

        st.markdown(f"**符合條件：{total} 篇**")

        page = st.number_input("頁碼", min_value=1, max_value=total_pages,
                               value=1, step=1, label_visibility="collapsed")
        st.caption(f"第 {page} / {total_pages} 頁（每頁 {PAGE_SIZE} 篇）")

        start_idx = (page - 1) * PAGE_SIZE
        page_articles = pool[start_idx: start_idx + PAGE_SIZE]

        # --- 文章列表 ---
        for article in page_articles:
            score = article.get("sentiment_score")
            if score is not None:
                icon = "🟢" if score > 0.2 else ("🔴" if score < -0.2 else "⚪")
            else:
                icon = "❓"

            is_news = article.get("source") == "cnyes"
            src_tag = "📰" if is_news else "💬"
            score_str = f"{score:+.2f}" if score is not None else "N/A"
            dt = str(article.get("datetime", ""))[:16]

            with st.expander(f"{icon} {src_tag} [{score_str}] {article.get('title', '無標題')}"):
                meta_col, _ = st.columns([3, 1])
                with meta_col:
                    st.markdown(
                        f"**時間:** {dt} ｜ "
                        f"**來源:** {'鉅亨新聞' if is_news else 'PTT'} ｜ "
                        f"**情緒:** {score_str} ｜ "
                        f"**標籤:** {article.get('sentiment_label', 'N/A')}"
                    )
                    if not is_news:
                        st.markdown(
                            f"**分類:** {article.get('category', 'N/A')} ｜ "
                            f"**作者:** {article.get('author', 'N/A')}"
                        )
                    reasoning = article.get("sentiment_reasoning", "")
                    if reasoning:
                        st.info(f"💡 {reasoning}")
                    content = article.get("content", "")
                    if content:
                        st.caption(
                            content[:400] + ("..." if len(content) > 400 else ""))


# ================================================================== #
#  Tab 5: 即時情緒分析
# ================================================================== #
with tab5:
    st.subheader("即時情緒分析")
    st.caption("輸入任何標題，AI 即時判斷對台積電股價的情緒影響")

    # 範例按鈕
    st.markdown("**快速試玩：**")
    examples = [
        "台積電法說會毛利率超預期 外資大買三萬張",
        "台積電下修Q2展望 庫存去化不如預期",
        "護國神山要噴了 大摩喊目標價1200",
        "台積電被美國制裁風險升溫 外資提款機",
        "台積電CoWoS產能滿載到2026 AI需求爆發",
    ]
    btn_cols = st.columns(len(examples))
    selected_example = ""
    for i, (col, ex) in enumerate(zip(btn_cols, examples)):
        if col.button(f"例 {i+1}", key=f"ex_{i}", use_container_width=True, help=ex):
            selected_example = ex

    user_input = st.text_input(
        "輸入標題：",
        value=selected_example,
        placeholder="例如：台積電法說會毛利率超預期 外資大買三萬張",
    )

    if st.button("分析情緒", type="primary", use_container_width=False) and user_input:
        with st.spinner("AI 分析中..."):
            try:
                from src.sentiment.llm_analyzer import SentimentAnalyzer
                analyzer = SentimentAnalyzer()
                result = analyzer.analyze(
                    f"realtime_{hash(user_input)}", user_input)

                if result:
                    score = result["score"]
                    normalized = (score + 1) / 2

                    if score > 0.6:
                        emoji, color, label_zh = "🚀", "#26a69a", "極度利多"
                    elif score > 0.2:
                        emoji, color, label_zh = "🟢", "#66bb6a", "利多"
                    elif score > -0.2:
                        emoji, color, label_zh = "⚪", "gray", "中性"
                    elif score > -0.6:
                        emoji, color, label_zh = "🔴", "#ef9a9a", "利空"
                    else:
                        emoji, color, label_zh = "💀", "#ef5350", "極度利空"

                    r1, r2 = st.columns([1, 2])
                    with r1:
                        st.metric("情緒分數", f"{score:+.2f}", label_zh)
                        st.progress(normalized, text=f"利空 ◄ {score:+.2f} ► 利多")
                    with r2:
                        st.markdown(f"### {emoji} {label_zh}")
                        st.info(f"**判斷理由：** {result['reasoning']}")
                else:
                    st.error("分析失敗，請確認 API Key 已設定")
            except Exception as e:
                st.error(f"錯誤: {e}")


# === Footer ===
st.divider()
st.caption("⚠️ 本專案僅供學術研究與技術展示，不構成任何投資建議。")

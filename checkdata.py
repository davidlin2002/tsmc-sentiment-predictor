from collections import Counter
import json

articles = []
with open("data/processed/ptt_with_sentiment.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        articles.append(json.loads(line))

has_score = [a for a in articles if a.get("sentiment_score") is not None]
no_score = [a for a in articles if a.get("sentiment_score") is None]

print(f"總文章數: {len(articles)}")
print(f"有情緒分數: {len(has_score)}")
print(f"沒有情緒分數: {len(no_score)}")

# 看每月分布
months = Counter()
for a in has_score:
    dt = a.get("datetime", "")
    if dt:
        months[dt[:7]] += 1

for month, count in sorted(months.items()):
    print(f"  {month}: {count} 篇有分數")

import finnhub
from transformers import pipeline
from datetime import datetime
import os
from dotenv import load_dotenv

# Initialize client and sentiment pipeline once
load_dotenv()
API_KEY = os.getenv("API_KEY")
finnhub_client = finnhub.Client(api_key=API_KEY)

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

sentiment_map = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

def get_avg_sentiment(symbol: str, date: str):
    news = finnhub_client.company_news(symbol, _from=date, to=date)
    articles = news[:10]
    scores = []

    if not articles:
        return None  # no articles available

    for item in articles:
        sentiment = sentiment_pipeline(item['summary'])[0]
        label = sentiment['label'].lower()
        score = sentiment_map.get(label, 1)  # default to neutral
        scores.append(score)
        #print(f'{i}/{len(articles)} Processed')

    avg_score = sum(scores) / len(scores)
    return avg_score
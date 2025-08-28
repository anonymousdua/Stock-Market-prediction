import finnhub
from transformers import pipeline
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import time
import pickle
from pathlib import Path
import streamlit as st

# Initialize client and sentiment pipeline once
load_dotenv()
API_KEY = os.getenv("API_KEY")
finnhub_client = finnhub.Client(api_key=API_KEY)

# Cache the sentiment model to avoid reloading
@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

sentiment_map = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

# Rate limiting tracker
class RateLimiter:
    def __init__(self, max_calls=60, time_window=60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    def wait_if_needed(self):
        now = time.time()
        # Remove calls older than time_window
        self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
        
        if len(self.calls) >= self.max_calls:
            # Calculate how long to wait
            oldest_call = min(self.calls)
            wait_time = self.time_window - (now - oldest_call) + 1
            if wait_time > 0:
                time.sleep(wait_time)
                # Clean up again after waiting
                now = time.time()
                self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
        
        self.calls.append(now)

# Global rate limiter
rate_limiter = RateLimiter(max_calls=55, time_window=60)  # Leave some buffer

def get_cached_sentiment(symbol: str, date: str):
    """Check if sentiment is already cached"""
    cache_dir = Path("sentiment_cache")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{symbol}_{date}.pkl"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            # If cache file is corrupted, remove it
            cache_file.unlink()
    
    return None

def save_sentiment_cache(symbol: str, date: str, sentiment_score: float):
    """Save sentiment to cache"""
    cache_dir = Path("sentiment_cache")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{symbol}_{date}.pkl"
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(sentiment_score, f)
    except Exception as e:
        print(f"Failed to cache sentiment for {symbol} {date}: {e}")

def get_avg_sentiment(symbol: str, date: str):
    """Get average sentiment with caching and rate limiting"""
    
    # Check cache first
    cached_sentiment = get_cached_sentiment(symbol, date)
    if cached_sentiment is not None:
        return cached_sentiment
    
    try:
        # Apply rate limiting before API call
        rate_limiter.wait_if_needed()
        
        # Try to get news for the specific date
        news = finnhub_client.company_news(symbol, _from=date, to=date)
        
        # If no news on specific date, expand search window (but limit to avoid too many API calls)
        if not news:
            # Try previous day
            prev_date = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
            rate_limiter.wait_if_needed()
            news = finnhub_client.company_news(symbol, _from=prev_date, to=date)
        
        if not news:
            # Default to neutral if no news found
            save_sentiment_cache(symbol, date, 1.0)
            return 1.0
        
        # Limit to top 5 articles to reduce processing time
        articles = news[:5]
        scores = []
        
        sentiment_pipeline = load_sentiment_model()
        
        for item in articles:
            try:
                sentiment = sentiment_pipeline(item['summary'])[0]
                label = sentiment['label'].lower()
                score = sentiment_map.get(label, 1)  # default to neutral
                scores.append(score)
            except Exception as e:
                print(f"Error processing article sentiment: {e}")
                scores.append(1)  # default to neutral
        
        if scores:
            avg_score = sum(scores) / len(scores)
        else:
            avg_score = 1.0  # default to neutral
        
        # Cache the result
        save_sentiment_cache(symbol, date, avg_score)
        return avg_score
    
    except Exception as e:
        print(f"Error getting sentiment for {symbol} on {date}: {e}")
        # Cache the neutral result to avoid repeated failures
        save_sentiment_cache(symbol, date, 1.0)
        return 1.0

def batch_get_sentiments(symbol: str, dates: list):
    """Get sentiments for multiple dates efficiently"""
    results = {}
    
    # Check cache for all dates first
    uncached_dates = []
    for date in dates:
        cached = get_cached_sentiment(symbol, date)
        if cached is not None:
            results[date] = cached
        else:
            uncached_dates.append(date)
    
    # Process uncached dates
    for date in uncached_dates:
        sentiment = get_avg_sentiment(symbol, date)
        results[date] = sentiment
        
        # Add a small delay to be extra safe with rate limiting
        time.sleep(0.1)
    
    return results
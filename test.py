"""import yfinance as yf
import requests
from bs4 import BeautifulSoup
import time

def get_article_content(url):
    
    Fetch full article content from URL
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'advertisement']):
            element.decompose()
        
        # Try different content selectors (ordered by priority)
        content_selectors = [
            # Yahoo Finance specific
            '.caas-body',
            '.article-body',
            
            # Common article selectors
            'article',
            '.story-body',
            '.entry-content',
            '.post-content',
            '.article-content',
            '.content',
            
            # Fallback selectors
            'main',
            '.main-content',
            '#main-content'
        ]
        
        article_content = None
        for selector in content_selectors:
            content_element = soup.select_one(selector)
            if content_element:
                article_content = content_element
                break
        
        if article_content:
            # Extract text and clean it up
            text = article_content.get_text(separator='\n', strip=True)
            
            # Clean up excessive whitespace
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            clean_text = '\n'.join(lines)
            
            return clean_text
        else:
            return "Could not extract article content - content selector not found"
            
    except requests.exceptions.RequestException as e:
        return f"Error fetching article: Network error - {str(e)}"
    except Exception as e:
        return f"Error parsing article: {str(e)}"

# Main code
ticker = yf.Ticker("AAPL")
news_items = ticker.news

for i, item in enumerate(news_items, 1):
    print(f"=== Article {i} ===")
    
    # Debug: Print the item structure to see available fields
    print("Available keys in item:", list(item.keys()))
    
    # Try different ways to extract URL
    url = None
    
    # Method 1: Direct keys
    if 'canonicalUrl' in item and isinstance(item['canonicalUrl'], dict):
        url = item['canonicalUrl'].get('url')
    
    # Method 2: clickThroughUrl
    if not url and 'clickThroughUrl' in item and isinstance(item['clickThroughUrl'], dict):
        url = item['clickThroughUrl'].get('url')
    
    # Method 3: Check if there's a nested structure
    if not url and 'content' in item:
        content_data = item['content']
        if isinstance(content_data, dict):
            print("Content keys:", list(content_data.keys()))
    
    # Method 4: Look for any key containing 'url' or 'link'
    if not url:
        for key, value in item.items():
            if 'url' in key.lower() or 'link' in key.lower():
                if isinstance(value, dict) and 'url' in value:
                    url = value['url']
                    break
                elif isinstance(value, str) and value.startswith('http'):
                    url = value
                    break
    
    print(f"Extracted URL: {url}")
    
    if url:
        print("\nFetching content...")
        content = get_article_content(url)
        print("Content:")
        print(content)
        print("\n" + "="*80 + "\n")
        
        # Add delay to be respectful to servers
        time.sleep(2)
    else:
        print("No URL could be extracted from this article")
        print("Item structure:", item)
        print("\n" + "="*80 + "\n")

"""


import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import time
import re
from urllib.parse import urljoin, quote

class FreeStockNewsFetcher:
    def __init__(self):
        """
        Initialize the news fetcher with free sources
        No API key required!
        """
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def fetch_yahoo_finance_news(self, ticker, limit=10):
        """
        Fetch news from Yahoo Finance (free, no API key needed)
        
        Args:
            ticker (str): Stock ticker symbol
            limit (int): Number of articles to fetch
        
        Returns:
            list: List of news articles
        """
        articles = []
        
        try:
            url = f"https://finance.yahoo.com/quote/{ticker.upper()}/news"
            print(f"Fetching Yahoo Finance news for {ticker.upper()}...")
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find news articles
            news_items = soup.find_all('li', class_='js-stream-content')
            
            for item in news_items[:limit]:
                article = {}
                
                # Title and link
                title_elem = item.find('h3')
                if title_elem:
                    link_elem = title_elem.find('a')
                    if link_elem:
                        article['title'] = link_elem.get_text(strip=True)
                        article['url'] = urljoin('https://finance.yahoo.com', link_elem.get('href', ''))
                
                # Source and time
                source_elem = item.find('div', class_='C(#959595)')
                if source_elem:
                    source_text = source_elem.get_text(strip=True)
                    # Split source and time
                    parts = source_text.split('•')
                    if len(parts) >= 2:
                        article['source'] = parts[0].strip()
                        article['time'] = parts[1].strip()
                    else:
                        article['source'] = source_text
                
                # Summary
                summary_elem = item.find('p')
                if summary_elem:
                    article['summary'] = summary_elem.get_text(strip=True)
                
                if article.get('title'):
                    articles.append(article)
            
        except Exception as e:
            print(f"Error fetching Yahoo Finance news: {e}")
        
        return articles
    
    def fetch_marketwatch_news(self, ticker, limit=10):
        """
        Fetch news from MarketWatch (free, no API key needed)
        
        Args:
            ticker (str): Stock ticker symbol
            limit (int): Number of articles to fetch
        
        Returns:
            list: List of news articles
        """
        articles = []
        
        try:
            # Search for ticker news on MarketWatch
            search_url = f"https://www.marketwatch.com/search?q={ticker.upper()}&m=Keyword&rpp=25&mp=806&bd=false&rs=true"
            print(f"Fetching MarketWatch news for {ticker.upper()}...")
            
            response = self.session.get(search_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find news articles
            news_items = soup.find_all('div', class_='searchresult')
            
            for item in news_items[:limit]:
                article = {}
                
                # Title and link
                title_elem = item.find('a', class_='searchresult__headline')
                if title_elem:
                    article['title'] = title_elem.get_text(strip=True)
                    article['url'] = urljoin('https://www.marketwatch.com', title_elem.get('href', ''))
                
                # Time
                time_elem = item.find('span', class_='searchresult__timestamp')
                if time_elem:
                    article['time'] = time_elem.get_text(strip=True)
                
                # Summary
                summary_elem = item.find('p', class_='searchresult__summary')
                if summary_elem:
                    article['summary'] = summary_elem.get_text(strip=True)
                
                article['source'] = 'MarketWatch'
                
                if article.get('title'):
                    articles.append(article)
                    
        except Exception as e:
            print(f"Error fetching MarketWatch news: {e}")
        
        return articles
    
    def fetch_seeking_alpha_news(self, ticker, limit=10):
        """
        Fetch news from Seeking Alpha (free, no API key needed)
        
        Args:
            ticker (str): Stock ticker symbol
            limit (int): Number of articles to fetch
        
        Returns:
            list: List of news articles
        """
        articles = []
        
        try:
            url = f"https://seekingalpha.com/symbol/{ticker.upper()}/news"
            print(f"Fetching Seeking Alpha news for {ticker.upper()}...")
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find news articles
            news_items = soup.find_all('article')[:limit]
            
            for item in news_items:
                article = {}
                
                # Title and link
                title_elem = item.find('h3') or item.find('h4')
                if title_elem:
                    link_elem = title_elem.find('a')
                    if link_elem:
                        article['title'] = link_elem.get_text(strip=True)
                        article['url'] = urljoin('https://seekingalpha.com', link_elem.get('href', ''))
                
                # Time
                time_elem = item.find('time')
                if time_elem:
                    article['time'] = time_elem.get('datetime') or time_elem.get_text(strip=True)
                
                # Summary
                summary_elem = item.find('p')
                if summary_elem:
                    article['summary'] = summary_elem.get_text(strip=True)
                
                article['source'] = 'Seeking Alpha'
                
                if article.get('title'):
                    articles.append(article)
                    
        except Exception as e:
            print(f"Error fetching Seeking Alpha news: {e}")
        
        return articles
    
    def fetch_all_sources(self, ticker, limit_per_source=5):
        """
        Fetch news from all available sources
        
        Args:
            ticker (str): Stock ticker symbol
            limit_per_source (int): Number of articles per source
        
        Returns:
            list: Combined list of news articles
        """
        all_articles = []
        
        # Fetch from Yahoo Finance
        yahoo_articles = self.fetch_yahoo_finance_news(ticker, limit_per_source)
        all_articles.extend(yahoo_articles)
        
        time.sleep(1)  # Be respectful to servers
        
        # Fetch from MarketWatch
        marketwatch_articles = self.fetch_marketwatch_news(ticker, limit_per_source)
        all_articles.extend(marketwatch_articles)
        
        time.sleep(1)  # Be respectful to servers
        
        # Fetch from Seeking Alpha
        seeking_alpha_articles = self.fetch_seeking_alpha_news(ticker, limit_per_source)
        all_articles.extend(seeking_alpha_articles)
        
        # Remove duplicates based on title
        seen_titles = set()
        unique_articles = []
        for article in all_articles:
            title = article.get('title', '').lower()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_articles.append(article)
        
        return unique_articles[:10]  # Return top 10 unique articles
    
    def display_articles(self, articles, ticker):
        """
        Display formatted news articles
        
        Args:
            articles (list): List of news articles
            ticker (str): Stock ticker symbol
        """
        if not articles:
            print(f"No articles found for {ticker}")
            return
        
        print(f"\n{'='*60}")
        print(f"NEWS ARTICLES FOR {ticker.upper()}")
        print(f"{'='*60}")
        
        for i, article in enumerate(articles, 1):
            print(f"\n{i}. {article.get('title', 'No Title')}")
            print(f"   Source: {article.get('source', 'Unknown')}")
            
            time_info = article.get('time', 'Unknown')
            print(f"   Published: {time_info}")
            
            url = article.get('url', 'No URL')
            print(f"   URL: {url}")
            
            # Display summary if available
            summary = article.get('summary', '')
            if summary:
                # Truncate long summaries
                summary = summary[:200] + "..." if len(summary) > 200 else summary
                print(f"   Summary: {summary}")
            
            print("-" * 60)
    
    def save_to_file(self, articles, ticker, filename=None):
        """
        Save articles to a JSON file
        
        Args:
            articles (list): List of news articles
            ticker (str): Stock ticker symbol
            filename (str): Optional filename
        """
        if not articles:
            print("No articles to save")
            return
        
        if not filename:
            date_str = datetime.now().strftime("%Y%m%d")
            filename = f"{ticker.lower()}_news_{date_str}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'ticker': ticker.upper(),
                    'fetch_date': datetime.now().isoformat(),
                    'article_count': len(articles),
                    'articles': articles
                }, f, indent=2, ensure_ascii=False)
            
            print(f"\nArticles saved to: {filename}")
            
        except IOError as e:
            print(f"Error saving file: {e}")

def main():
    """
    Main function to demonstrate usage
    """
    # Initialize the news fetcher (no API key needed!)
    fetcher = FreeStockNewsFetcher()
    
    # Get ticker from user input
    ticker = input("Enter stock ticker (e.g., AAPL, TSLA, MSFT): ").strip()
    
    if not ticker:
        print("Please enter a valid ticker symbol")
        return
    
    # Choose source
    print("\nChoose news source:")
    print("1. Yahoo Finance")
    print("2. MarketWatch")  
    print("3. Seeking Alpha")
    print("4. All sources (combined)")
    
    choice = input("Enter choice (1-4): ").strip()
    
    # Fetch news articles based on choice
    if choice == "1":
        articles = fetcher.fetch_yahoo_finance_news(ticker, 10)
    elif choice == "2":
        articles = fetcher.fetch_marketwatch_news(ticker, 10)
    elif choice == "3":
        articles = fetcher.fetch_seeking_alpha_news(ticker, 10)
    elif choice == "4":
        articles = fetcher.fetch_all_sources(ticker, 4)  # 4 per source = ~12 total
    else:
        print("Invalid choice, using all sources...")
        articles = fetcher.fetch_all_sources(ticker, 4)
    
    # Display articles
    fetcher.display_articles(articles, ticker)
    
    # Ask if user wants to save to file
    if articles:
        save_choice = input("\nSave articles to file? (y/n): ").strip().lower()
        if save_choice == 'y':
            fetcher.save_to_file(articles, ticker)

if __name__ == "__main__":
    main()


    
import os
import yfinance as yf
from groq import Groq
import logging

logger = logging.getLogger("Argus.LLM")

def get_recent_news_summary(ticker):
    """Fetch the latest news headlines from yfinance."""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news:
            return "No recent news available."
        
        headlines = []
        for item in news[:4]:  # Top 4 news items
            title = item.get("title", "")
            publisher = item.get("publisher", "")
            if title:
                headlines.append(f"- {title} ({publisher})")
        return "\n".join(headlines)
    except Exception as e:
        logger.warning(f"Failed to fetch news for {ticker}: {e}")
        return "News fetch failed."

def generate_ai_thesis(ticker, score, reasons, groq_api_key):
    """Call Groq to generate a 3-bullet thesis."""
    if not groq_api_key:
        return "GROQ API Key missing. Skipping AI analysis."
    
    try:
        news_summary = get_recent_news_summary(ticker)
        client = Groq(api_key=groq_api_key)
        
        prompt = f"""
        You are an expert quantitative hedge fund analyst. Argus screener just flagged {ticker} with a high score of {score}/100.
        The technical and fundamental reasons for this score are: {', '.join(reasons)}.
        
        Here are the most recent news headlines for {ticker}:
        {news_summary}
        
        Provide a concise, 3-sentence summary of why this stock is a good buy. 
        Focus on integrating the technical score, fundamental momentum, and the current news sentiment.
        Keep it professional, brief, and highly insightful. Start directly with the thesis, no pleasantries.
        """
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=250,
        )
        
        thesis = response.choices[0].message.content.strip()
        return thesis
    except Exception as e:
        logger.error(f"Groq AI analysis failed for {ticker}: {e}")
        return "AI analysis failed due to an API error."

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
_CATALYST_TAGS = [
    "FDA", "defense_contract", "govt_award", "partnership", "ai_robotics",
    "clinical_trial", "earnings_beat", "merger_acquisition",
]

_CATALYST_TAG_WEIGHTS = {
    "FDA": 4,
    "defense_contract": 4,
    "govt_award": 4,
    "partnership": 3,
    "ai_robotics": 3,
    "clinical_trial": 3,
    "earnings_beat": 2,
    "merger_acquisition": 2,
}


def get_llm_catalyst_analysis(ticker, groq_api_key, fmp_context=None):
    """
    Fetch news and call Groq for structured catalyst analysis.
    Returns dict: {"sentiment": int, "catalysts": list[str], "urgency": str}
    sentiment is -10 to +10.
    """
    if not groq_api_key:
        return {"sentiment": 0, "catalysts": [], "urgency": "longterm"}

    try:
        import json as _json
        news_summary = get_recent_news_summary(ticker)
        if not news_summary or news_summary == "No recent news available." or "Error" in news_summary:
            return {"sentiment": 0, "catalysts": [], "urgency": "longterm"}

        fmp_context_str = ""
        if fmp_context:
            fmp_context_str = f"\nAdditional context: {fmp_context}"

        client = Groq(api_key=groq_api_key)
        tags_str = ", ".join(_CATALYST_TAGS)
        prompt = (
            f"You are a quantitative analyst. Analyze these news headlines for {ticker}.\n"
            f"Headlines: {news_summary}"
            f"{fmp_context_str}\n\n"
            f"Return ONLY valid JSON (no markdown):\n"
            f'{{\"sentiment\": <integer -10 to +10>, \"catalysts\": [<list of tags from: {tags_str}>], '
            f'\"urgency\": \"<imminent|near|longterm>\"}}\n\n'
            f"Rules:\n"
            f"- sentiment: overall news tone (-10=very bearish, 0=neutral, +10=very bullish)\n"
            f"- catalysts: only include if clearly evidenced in headlines\n"
            f"- urgency: imminent if event is <30 days away, near if <90 days, longterm otherwise"
        )

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.2,
            max_tokens=120,
        )

        raw = response.choices[0].message.content.strip()
        result = _json.loads(raw)
        sentiment = int(result.get("sentiment", 0))
        sentiment = max(-10, min(10, sentiment))
        catalysts = [t for t in result.get("catalysts", []) if t in _CATALYST_TAGS]
        urgency = result.get("urgency", "longterm")
        if urgency not in ("imminent", "near", "longterm"):
            urgency = "longterm"
        return {"sentiment": sentiment, "catalysts": catalysts, "urgency": urgency}
    except Exception as e:
        logger.warning(f"LLM catalyst analysis failed for {ticker}: {e}")
        return {"sentiment": 0, "catalysts": [], "urgency": "longterm"}


def get_sentiment_score(ticker, groq_api_key):
    """
    Backward-compatible wrapper. Returns integer sentiment from -10 to +10.
    """
    result = get_llm_catalyst_analysis(ticker, groq_api_key)
    return result["sentiment"]
def generate_ai_thesis(ticker, score, reasons, groq_api_key):
    """Call Groq for structured catalyst analysis.

    Returns a 3-line formatted string:
      💡 Theme: <sector/theme driving the stock>
      🚀 Catalyst: <specific forward-looking reason to run>
      ⚠️ Risk: <key bear case> <strength emoji>

    Returns None on failure so callers can skip display cleanly.
    """
    if not groq_api_key:
        return None

    try:
        import json as _json
        news_summary = get_recent_news_summary(ticker)
        client = Groq(api_key=groq_api_key)

        reasons_str = ", ".join(reasons[:10]) if reasons else "no specific signals"

        prompt = (
            f"You are a stock analyst. Analyze {ticker} which scored {score}/100 on a quant screener.\n"
            f"Screener signals: {reasons_str}\n"
            f"Recent news: {news_summary}\n\n"
            f"Return ONLY valid JSON (no markdown, no explanation):\n"
            f'{{\"sector_thesis\": \"<5-8 word phrase naming the theme driving this stock>\", '
            f'\"top_catalyst\": \"<one forward-looking reason this could run — cite specific evidence>\", '
            f'\"risk_flag\": \"<one key bear case specific to this stock>\", '
            f'\"signal_strength\": \"<strong|moderate|weak>\"}}\n\n'
            f"Rules:\n"
            f"- sector_thesis: e.g. 'Government AI contract pipeline', 'Railroad automation mandate'\n"
            f"- top_catalyst: must be specific, not generic — cite news headline or screener signal\n"
            f"- risk_flag: must be specific to this stock, not boilerplate\n"
            f"- signal_strength: strong if 3+ signals align, moderate if 2, weak if 1"
        )

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.2,
            max_tokens=150,
        )

        raw = response.choices[0].message.content.strip()
        data = _json.loads(raw)

        theme    = (data.get("sector_thesis") or "").strip()
        catalyst = (data.get("top_catalyst") or "").strip()
        risk     = (data.get("risk_flag") or "").strip()
        strength = (data.get("signal_strength") or "moderate").strip().lower()

        if not theme and not catalyst:
            return None

        strength_emoji = {"strong": "🟢", "moderate": "🟡", "weak": "🔴"}.get(strength, "🟡")
        lines = []
        if theme:
            lines.append(f"💡 Theme: {theme}")
        if catalyst:
            lines.append(f"🚀 Catalyst: {catalyst}")
        if risk:
            lines.append(f"⚠️ Risk: {risk} {strength_emoji}")
        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Groq AI analysis failed for {ticker}: {e}")
        return None

#!/usr/bin/env python3
"""Backtest harness for Argus scoring.

Usage:
  python backtest.py --horizon 63 --threshold 70
  python backtest.py --horizon 126 --threshold 75 --weights config/weights_runner.json

Reads argus_feature_history.csv and argus_results_history.csv.
Fetches forward price data from yfinance for each pick.
Outputs: hit rate table, top performers, median return.
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


def load_history():
    feat_path = "data/argus_feature_history.csv"
    hist_path = "data/argus_results_history.csv"
    if not os.path.exists(feat_path):
        print(f"ERROR: {feat_path} not found. Run a scan first.")
        sys.exit(1)
    feat_df = pd.read_csv(feat_path)
    hist_df = pd.read_csv(hist_path) if os.path.exists(hist_path) else pd.DataFrame()
    return feat_df, hist_df


def fetch_forward_return(ticker: str, start_date: str, horizon_days: int) -> float | None:
    """Fetch forward return from start_date over horizon_days calendar days."""
    try:
        from datetime import datetime, timedelta
        start = pd.to_datetime(start_date)
        end   = start + timedelta(days=horizon_days + 30)  # buffer for weekends
        hist  = yf.Ticker(ticker).history(start=start.strftime("%Y-%m-%d"),
                                          end=end.strftime("%Y-%m-%d"), progress=False)
        if hist.empty or len(hist) < 5:
            return None
        entry_price = hist["Close"].iloc[0]
        # Find the row closest to horizon_days trading days out
        target_idx  = min(horizon_days, len(hist) - 1)
        exit_price  = hist["Close"].iloc[target_idx]
        return float((exit_price / entry_price) - 1)
    except Exception:
        return None


def run_backtest(horizon_days: int = 63, threshold: int = 70, weights_file: str = None,
                 min_samples: int = 10) -> dict:
    feat_df, hist_df = load_history()

    if feat_df.empty:
        return {"error": "No feature history available"}

    # Normalise date column
    date_col = "scan_date" if "scan_date" in feat_df.columns else feat_df.columns[0]
    feat_df[date_col] = pd.to_datetime(feat_df[date_col])
    feat_df = feat_df.sort_values(date_col)

    # Only use picks old enough to have matured
    from datetime import datetime, timedelta
    cutoff = datetime.now() - timedelta(days=horizon_days + 5)
    mature = feat_df[feat_df[date_col] < cutoff].copy()

    if len(mature) < min_samples:
        return {"error": f"Only {len(mature)} matured samples (need {min_samples}). Run more scans first."}

    # Filter by score threshold
    picks = mature[mature["score"] >= threshold].copy()
    if picks.empty:
        return {"error": f"No picks with score >= {threshold} in history"}

    # Deduplicate (take first occurrence of each ticker per week)
    picks["week"] = picks[date_col].dt.to_period("W")
    picks = picks.drop_duplicates(subset=["ticker", "week"])

    print(f"Backtesting {len(picks)} picks (score>={threshold}, horizon={horizon_days}d)...")

    # Fetch forward returns
    results = []
    for _, row in picks.iterrows():
        fwd = fetch_forward_return(str(row["ticker"]), str(row[date_col].date()), horizon_days)
        if fwd is not None:
            results.append({
                "ticker":      row["ticker"],
                "scan_date":   str(row[date_col].date()),
                "score":       row["score"],
                "fwd_return":  round(fwd * 100, 1),
                "sector":      row.get("sector", "Unknown"),
            })

    if not results:
        return {"error": "Could not fetch any forward returns"}

    df = pd.DataFrame(results)
    hit_rate_50pct = (df["fwd_return"] >= 50).mean() * 100
    hit_rate_100pct = (df["fwd_return"] >= 100).mean() * 100
    hit_rate_200pct = (df["fwd_return"] >= 200).mean() * 100
    hit_rate_pos    = (df["fwd_return"] > 0).mean() * 100
    median_ret      = df["fwd_return"].median()
    max_ret         = df["fwd_return"].max()
    min_ret         = df["fwd_return"].min()

    top10 = df.nlargest(10, "fwd_return")[["ticker", "scan_date", "score", "fwd_return", "sector"]]

    summary = {
        "n_picks":          len(df),
        "horizon_days":     horizon_days,
        "threshold":        threshold,
        "hit_rate_pos_pct": round(hit_rate_pos, 1),
        "hit_rate_50pct":   round(hit_rate_50pct, 1),
        "hit_rate_100pct":  round(hit_rate_100pct, 1),
        "hit_rate_200pct":  round(hit_rate_200pct, 1),
        "median_return_pct": round(median_ret, 1),
        "max_return_pct":    round(max_ret, 1),
        "min_return_pct":    round(min_ret, 1),
        "top_performers":    top10.to_dict("records"),
        "sector_breakdown":  df.groupby("sector")["fwd_return"].median().round(1).to_dict(),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Argus backtest harness")
    parser.add_argument("--horizon",   type=int,   default=63,   help="Forward return horizon in calendar days")
    parser.add_argument("--threshold", type=int,   default=70,   help="Minimum score to include in backtest")
    parser.add_argument("--weights",   type=str,   default=None, help="Optional weights JSON file (for display)")
    parser.add_argument("--min-samples", type=int, default=10,   help="Minimum picks required to run")
    args = parser.parse_args()

    weights_info = ""
    if args.weights and os.path.exists(args.weights):
        with open(args.weights) as f:
            w = json.load(f)
        weights_info = f" [{w.get('description', args.weights)}]"

    print(f"\n{'='*60}")
    print(f"Argus Backtest{weights_info}")
    print(f"Horizon: {args.horizon}d | Min score: {args.threshold}")
    print(f"{'='*60}\n")

    result = run_backtest(args.horizon, args.threshold, args.weights, args.min_samples)

    if "error" in result:
        print(f"ERROR: {result['error']}")
        sys.exit(1)

    print(f"Picks analysed:   {result['n_picks']}")
    print(f"Win rate (>0%):   {result['hit_rate_pos_pct']}%")
    print(f"Hit rate (>50%):  {result['hit_rate_50pct']}%")
    print(f"Hit rate (>100%): {result['hit_rate_100pct']}%")
    print(f"Hit rate (>200%): {result['hit_rate_200pct']}%")
    print(f"Median return:    {result['median_return_pct']}%")
    print(f"Best pick:        {result['max_return_pct']}%")
    print(f"\nTop 10 Performers:")
    for r in result["top_performers"]:
        print(f"  {r['ticker']:8s} {r['scan_date']}  score={r['score']}  +{r['fwd_return']}%  ({r['sector']})")
    print(f"\nSector median returns:")
    for sector, ret in sorted(result["sector_breakdown"].items(), key=lambda x: -x[1]):
        print(f"  {sector:30s} {ret:+.1f}%")


if __name__ == "__main__":
    main()

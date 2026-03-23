#!/usr/bin/env python3
"""CLI wrapper — run the quant scorer backtester.

Usage:
    python backtest.py                          # defaults: 90d, top 10 symbols
    python backtest.py --days 180 --symbols 15  # 6 months, 15 symbols
    python backtest.py --symbols-list ETH,SOL,XRP --days 60
    python backtest.py --min-score 55 --min-rr 1.5  # test lower thresholds
    python backtest.py --min-score 70 --min-rr 3.0  # test tighter filters
"""
import asyncio
import sys
from pathlib import Path

# Ensure backend is importable
sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

from app.services.backtester import main

if __name__ == "__main__":
    import argparse, logging

    parser = argparse.ArgumentParser(description="Backtest the quant scorer strategy")
    parser.add_argument("--days", type=int, default=90, help="Days of history (default: 90)")
    parser.add_argument("--symbols", type=int, default=10, help="Top N symbols by volume (default: 10)")
    parser.add_argument("--balance", type=float, default=10_000.0, help="Starting balance (default: 10000)")
    parser.add_argument("--min-score", type=float, default=None, help="Override MIN_QUANT_SCORE")
    parser.add_argument("--min-rr", type=float, default=None, help="Override MIN_REWARD_RISK_RATIO")
    parser.add_argument("--max-positions", type=int, default=5, help="Max concurrent positions (default: 5)")
    parser.add_argument("--symbols-list", type=str, default=None,
                        help="Comma-separated symbols (e.g. ETH,SOL,XRP) — overrides --symbols")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    asyncio.run(main(
        days=args.days,
        n_symbols=args.symbols,
        balance=args.balance,
        min_score=args.min_score,
        min_rr=args.min_rr,
        max_positions=args.max_positions,
        symbols_list=args.symbols_list,
        save_path=args.output,
    ))

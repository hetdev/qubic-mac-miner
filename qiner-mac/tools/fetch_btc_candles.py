#!/usr/bin/env python3
"""Fetch hourly BTC candles and cache them to a CSV.
Usage:
    python3 fetch_btc_candles.py --start 2023-01-01 --hours 8761
    python3 fetch_btc_candles.py --start 2023-01-01 --end 2024-01-01 --out btc_candles_1h.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

HOUR_MS: int = 3_600_000

# Public REST kline endpoints (no auth). "close" is field index 4; open time is index 0 (ms).
SOURCES: dict[str, str] = {
    "binance": "https://api.binance.com/api/v3/klines"
}


@dataclass(frozen=True)
class Candle:
    open_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


def _to_ms(day: str) -> int:
    """Parse YYYY-MM-DD (UTC midnight) to epoch milliseconds."""
    dt = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _http_get_json(url: str, params: dict[str, str | int], retries: int = 5) -> list:
    full = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(full, headers={"User-Agent": "bpp9000-fetch/1.0"})
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            last = exc
            wait = 2 * (attempt + 1)
            print(f"  request failed ({exc}); retry {attempt + 1}/{retries} in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"HTTP GET failed after {retries} attempts for {full}: {last}")


def fetch_binance_1h(symbol: str, start_ms: int, end_ms: int) -> list[Candle]:
    """Fetch [start_ms, end_ms) hourly klines from Binance, paginating (max 1000/request)."""
    url = SOURCES["binance"]
    out: list[Candle] = []
    cursor = start_ms
    while cursor < end_ms:
        rows = _http_get_json(
            url,
            {"symbol": symbol, "interval": "1h", "startTime": cursor, "endTime": end_ms, "limit": 1000},
        )
        if not rows:
            break
        for r in rows:
            out.append(
                Candle(
                    open_time_ms=int(r[0]),
                    open=float(r[1]),
                    high=float(r[2]),
                    low=float(r[3]),
                    close=float(r[4]),
                    volume=float(r[5]),
                )
            )
        last_open = int(rows[-1][0])
        nxt = last_open + HOUR_MS
        if nxt <= cursor:  # no progress -> stop to avoid an infinite loop
            break
        cursor = nxt
        print(f"  fetched up to {datetime.fromtimestamp(last_open / 1000, timezone.utc):%Y-%m-%d %H:%M} "
              f"({len(out)} candles)", file=sys.stderr)
    return out


def find_gaps(candles: list[Candle]) -> list[tuple[int, int]]:
    """Return (index, gap_hours) for every break where consecutive open times differ by != 1h.

    A broad download may contain rare exchange-maintenance gaps; that is fine - the analyze/convert steps
    only require the chosen 8761-hour window to be gap-free, and they enforce it.
    """
    gaps: list[tuple[int, int]] = []
    for i in range(1, len(candles)):
        gap = candles[i].open_time_ms - candles[i - 1].open_time_ms
        if gap != HOUR_MS:
            gaps.append((i, gap // HOUR_MS))
    return gaps


def read_cache(path: Path) -> list[Candle]:
    if not path.exists():
        return []
    out: list[Candle] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            out.append(
                Candle(
                    open_time_ms=int(row["openTime"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )
    return out


def write_cache(path: Path, candles: list[Candle]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["openTime", "open", "high", "low", "close", "volume"])
        for c in candles:
            w.writerow([c.open_time_ms, c.open, c.high, c.low, c.close, c.volume])


def main() -> int:
    ap = argparse.ArgumentParser(description="Download + cache hourly BTC candles for the bpp9000 price task.")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--start", required=True, help="UTC start date YYYY-MM-DD")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--end", help="UTC end date YYYY-MM-DD (exclusive)")
    group.add_argument("--hours", type=int, help="number of hourly candles to fetch (e.g. 8761)")
    ap.add_argument("--out", default="btc_candles_1h.csv", help="cache CSV path")
    ap.add_argument("--force", action="store_true", help="re-download even if the cache already covers the range")
    args = ap.parse_args()

    start_ms = _to_ms(args.start)
    if args.hours is not None:
        end_ms = start_ms + args.hours * HOUR_MS
    else:
        end_ms = _to_ms(args.end)
    expected = (end_ms - start_ms) // HOUR_MS
    out_path = Path(args.out)

    if not args.force:
        cached = read_cache(out_path)
        covered = [c for c in cached if start_ms <= c.open_time_ms < end_ms]
        # Cache hit if the cache already spans the requested range (internal gaps are tolerated here).
        if covered and covered[0].open_time_ms <= start_ms and covered[-1].open_time_ms >= end_ms - HOUR_MS:
            print(f"cache hit: {out_path} already spans the range ({len(covered)} candles) - skipping download")
            return 0

    print(f"downloading up to {expected} hourly {args.symbol} candles from Binance "
          f"({args.start} +{expected}h)...", file=sys.stderr)
    candles = fetch_binance_1h(args.symbol, start_ms, end_ms)
    candles = [c for c in candles if start_ms <= c.open_time_ms < end_ms]
    if not candles:
        raise RuntimeError("no candles returned - check the date range / symbol")
    write_cache(out_path, candles)

    gaps = find_gaps(candles)
    first = datetime.fromtimestamp(candles[0].open_time_ms / 1000, timezone.utc)
    last = datetime.fromtimestamp(candles[-1].open_time_ms / 1000, timezone.utc)
    print(f"wrote {len(candles)} candles (expected {expected}) to {out_path} "
          f"[{first:%Y-%m-%d %H:%M} .. {last:%Y-%m-%d %H:%M} UTC]")
    if gaps:
        print(f"note: {len(gaps)} gap(s) in this range (analyze/convert will avoid windows spanning them):")
        for idx, hours in gaps[:10]:
            at = datetime.fromtimestamp(candles[idx].open_time_ms / 1000, timezone.utc)
            print(f"  missing ~{hours - 1}h before {at:%Y-%m-%d %H:%M} UTC")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

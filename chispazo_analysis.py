#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CHISPAZO LOTTERY ANALYZER
=========================

Statistical analysis of a Chispazo CSV dataset with the columns:

    CONCURSO,R1,R2,R3,R4,R5,FECHA

* Number universe: 01..28
* Each contest draws 5 distinct numbers
* FECHA format: DD/MM/YYYY
* The analysis window is the last N contests (default: 100), selected by the
  official CONCURSO number (largest contest IDs), not by file order.

USAGE
-----
    python chispazo_analysis.py --csv "Chispazo(1).csv"
    python chispazo_analysis.py --csv "Chispazo(1).csv" --last 100
    python chispazo_analysis.py --csv "Chispazo(1).csv" --top 20 --outdir reports

REQUIREMENTS
------------
    Python 3.9+
    pandas
    numpy

WHAT THE SCRIPT PRODUCES
------------------------
Report 1: Game compression with 9- and 10-number pools
    * Enumerates every pool that can possibly contain at least two complete
      draws, then scores it exactly against the analysis window.
    * "5-Hit" = number of contests in the window in which all five winning
      numbers are contained in the pool.
    * Table columns: K, pool size, numbers, 5-Hit, 4-Hit+, 3-Hit+,
      full-wheel ticket count, hotness and the contests where the 5-Hit hit.

Report 2: Exclusion patterns and recurrent blocks
    * Sequential blocks of 10..14 consecutive numbers.
    * Data-driven blocks built by greedy co-occurrence clustering.
    * Exact recurrence counts for blocks with 3+ occurrences, where a
      recurrence means that at least 4 of the 5 drawn numbers fall inside
      the block.
    * Dormant-number, dormant-pair, low-frequency-cluster and low-recurrence
      block tables, with exact counts, for use as descriptive exclusion
      filters.

IMPORTANT STATISTICAL NOTE
--------------------------
Lottery draws are assumed to be random and independent. Hot/cold numbers,
recurrent blocks and dormant patterns are descriptions of a finite sample.
They do *not* change the probability of future draws. Treat the exclusion
filters as hypotheses for backtesting, never as a guarantee.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUMBER_MIN = 1
NUMBER_MAX = 28
DRAW_SIZE = 5
DEFAULT_LAST_CONTESTS = 100
POOL_SIZES = (9, 10)
TOP_POOLS = 60

BLOCK_SIZES = tuple(range(10, 15))       # 10, 11, 12, 13, 14 numbers
MIN_RECURRENCE = 3                      # 3+ occurrences = "recurrent"
CLUSTER_SIZES = (10, 12)                # data-driven block sizes
CLUSTERS_PER_SIZE = 6
MIN_NEW_NUMBERS = 3                     # diversity constraint for clusters


# ---------------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------------
def load_dataset(csv_path: Path, last_n: int) -> tuple[pd.DataFrame, list[tuple[int, ...]]]:
    """Load, validate, sort and cut the Chispazo dataset.

    The last ``last_n`` contests are selected by the official CONCURSO number.
    FECHA is used only as a secondary tie-breaker and for display.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().upper() for c in df.columns]

    required = {"CONCURSO", "R1", "R2", "R3", "R4", "R5", "FECHA"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(sorted(missing))}")

    rows_in = len(df)

    # Coerce numeric columns and dates.
    for col in ("CONCURSO", "R1", "R2", "R3", "R4", "R5"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["FECHA"] = pd.to_datetime(
        df["FECHA"].astype(str).str.strip(),
        format="%d/%m/%Y",
        errors="coerce",
    )

    # Drop rows that cannot be validated.
    df = df.dropna(subset=["CONCURSO", "R1", "R2", "R3", "R4", "R5", "FECHA"])

    # All numbers must be integers in 01..28.
    for col in ("R1", "R2", "R3", "R4", "R5"):
        df = df[(df[col] % 1 == 0) & (df[col] >= NUMBER_MIN) & (df[col] <= NUMBER_MAX)]

    # Each draw must contain five distinct numbers.
    draw_cols = ["R1", "R2", "R3", "R4", "R5"]
    df = df[df[draw_cols].nunique(axis=1) == DRAW_SIZE]
    df["CONCURSO"] = df["CONCURSO"].astype(int)

    if df.empty:
        raise ValueError("No valid rows remain after validation.")

    # Sort chronologically by contest number; date resolves any duplicate contest.
    df = df.sort_values(["CONCURSO", "FECHA"], ascending=[True, True])

    # One row per contest. If a duplicated contest exists, keep the latest row.
    duplicated = int(df["CONCURSO"].duplicated().sum())
    if duplicated:
        df = df.drop_duplicates(subset="CONCURSO", keep="last")
        print(f"[warning] Removed {duplicated} duplicated CONCURSO row(s).")

    if len(df) < last_n:
        print(
            f"[warning] Requested the last {last_n} contests, but only "
            f"{len(df)} valid contests are available. Using all of them."
        )
        last_n = len(df)

    window = df.tail(last_n).reset_index(drop=True)
    draws = [
        tuple(sorted(int(v) for v in row))
        for row in window[draw_cols].to_numpy()
    ]

    rows_out = len(window)
    print("\nDATASET SUMMARY")
    print("-" * 72)
    print(f"CSV file ................ {csv_path}")
    print(f"Rows read ............... {rows_in}")
    print(f"Valid contests .......... {len(df)}")
    print(f"Dropped/invalid rows .... {rows_in - len(df)}")
    print(f"Analysis window ......... last {rows_out} contests")
    print(f"Contest range ........... {int(window['CONCURSO'].min())} .. {int(window['CONCURSO'].max())}")
    print(f"Date range .............. {window['FECHA'].min():%d/%m/%Y} .. {window['FECHA'].max():%d/%m/%Y}")
    print("-" * 72)

    return window, draws


def number_frequencies(draws: Iterable[Sequence[int]]) -> Counter:
    """Absolute frequency of every number in the analysis window."""
    freq: Counter = Counter()
    for draw in draws:
        freq.update(draw)
    return freq


def draw_masks(draws: Sequence[Sequence[int]]) -> list[int]:
    """Encode each draw as a bit mask (bit n represents number n)."""
    masks = []
    for draw in draws:
        mask = 0
        for n in draw:
            mask |= 1 << n
        masks.append(mask)
    return masks


def mask_to_numbers(mask: int) -> tuple[int, ...]:
    """Decode a bit mask into a sorted tuple of numbers."""
    return tuple(n for n in range(NUMBER_MIN, NUMBER_MAX + 1) if (mask >> n) & 1)


# ---------------------------------------------------------------------------
# Report 1: exact 5-hit pool search for 9- and 10-number pools
# ---------------------------------------------------------------------------
def candidate_pools_with_two_hits(draws: Sequence[Sequence[int]], k: int) -> set[int]:
    """Return all k-number pools that contain at least two complete draws.

    If a pool contains two draws A and B, it must be a superset of A union B.
    Therefore every pool with 2+ five-hits is generated by taking every pair
    of draws, keeping unions of size <= k, and adding arbitrary extra numbers
    until the pool reaches size k.

    This is exact for all pools with 2+ hits and is dramatically smaller than
    enumerating all C(28, k) possible pools (13,123,110 for k=10).
    """
    masks = draw_masks(draws)
    universe = list(range(NUMBER_MIN, NUMBER_MAX + 1))
    candidates: set[int] = set()

    for a, b in combinations(masks, 2):
        union = a | b
        union_size = union.bit_count()
        if union_size > k:
            continue

        free = [n for n in universe if not ((union >> n) & 1)]
        need = k - union_size
        for extra in combinations(free, need):
            pool = union
            for n in extra:
                pool |= 1 << n
            candidates.add(pool)

    return candidates


def score_pool_masks(pool_masks: Sequence[int], draw_masks_: Sequence[int]) -> np.ndarray:
    """Count, for every pool mask, how many draws are fully contained in it.

    The operation is vectorised in chunks so that large candidate sets remain
    memory-friendly.
    """
    pools = np.fromiter(pool_masks, dtype=np.int64)
    draws = np.asarray(draw_masks_, dtype=np.int64)
    hits = np.empty(len(pools), dtype=np.int16)

    chunk_size = 20_000
    for start in range(0, len(pools), chunk_size):
        block = pools[start:start + chunk_size]
        # Every draw must be a subset of the pool: (draw AND pool) == draw.
        contained = ((block[:, None] & draws[None, :]) == draws[None, :]).sum(axis=1)
        hits[start:start + chunk_size] = contained

    return hits


def fallback_single_hit_pools(
    draws: Sequence[Sequence[int]],
    k: int,
    freq: Counter,
    limit: int,
) -> list[tuple[int, tuple[int, ...]]]:
    """Build representative 1-hit pools when no pool reaches 2 hits.

    All such pools are tied at one hit. We therefore create a deterministic
    hot-number completion around each actual draw and return the hottest ones.
    """
    universe = list(range(NUMBER_MIN, NUMBER_MAX + 1))
    pools: dict[int, tuple[int, ...]] = {}

    for draw in draws:
        remaining = [n for n in universe if n not in draw]
        remaining.sort(key=lambda n: (-freq.get(n, 0), n))
        pool = tuple(sorted(draw + tuple(remaining[: k - DRAW_SIZE])))
        mask = sum(1 << n for n in pool)
        pools[mask] = pool

    ranked = sorted(
        pools.values(),
        key=lambda p: (-sum(freq.get(n, 0) for n in p), p),
    )
    return [(sum(1 << n for n in p), p) for p in ranked[:limit]]


def analyze_pools(
    window: pd.DataFrame,
    draws: Sequence[Sequence[int]],
    k: int,
    top_n: int,
) -> pd.DataFrame:
    """Find and score the best k-number pools by exact 5-hit count."""
    freq = number_frequencies(draws)
    masks = draw_masks(draws)
    expected_hits = len(draws) * math.comb(k, DRAW_SIZE) / math.comb(
        NUMBER_MAX, DRAW_SIZE
    )

    candidates = candidate_pools_with_two_hits(draws, k)

    if candidates:
        candidate_list = list(candidates)
        hits = score_pool_masks(candidate_list, masks)
        pool_array = np.fromiter(candidate_list, dtype=np.int64)

        # Tie-breakers: more 5-hits, then hotter pool, then stable numeric order.
        hotness = np.zeros(len(pool_array), dtype=np.int32)
        for n in range(NUMBER_MIN, NUMBER_MAX + 1):
            hotness += freq.get(n, 0) * ((pool_array >> n) & 1).astype(np.int32)

        order = np.lexsort((pool_array, -hotness, -hits))
        top_indices = order[: max(top_n, 1)]
        selected = [(int(pool_array[i]), int(hits[i])) for i in top_indices]
    else:
        # Extremely small windows can leave every pool at <= 1 hit.
        selected = fallback_single_hit_pools(draws, k, freq, top_n)

    rows = []
    for rank, (mask, hit5) in enumerate(selected, start=1):
        pool = mask_to_numbers(mask)
        overlap = [((mask & d).bit_count()) for d in masks]
        hit5_checked = sum(1 for v in overlap if v == DRAW_SIZE)
        hit4plus = sum(1 for v in overlap if v >= DRAW_SIZE - 1)
        hit3plus = sum(1 for v in overlap if v >= DRAW_SIZE - 2)

        hit_contests = [
            int(window.iloc[i]["CONCURSO"])
            for i, v in enumerate(overlap)
            if v == DRAW_SIZE
        ]

        rows.append(
            {
                "K": rank,
                "Pool size": k,
                "Numbers pool": "-".join(f"{n:02d}" for n in pool),
                "5-Hit": hit5_checked,
                "4-Hit+": hit4plus,
                "3-Hit+": hit3plus,
                "Wheel tickets": math.comb(k, DRAW_SIZE),
                "Hotness": sum(freq.get(n, 0) for n in pool),
                "Hit contests": ", ".join(str(c) for c in hit_contests) or "-",
                "_pool": pool,
                "_expected_5hit": expected_hits,
            }
        )

    result = pd.DataFrame(rows)
    result.attrs["expected_5hit"] = expected_hits
    result.attrs["candidate_count"] = len(candidates)
    return result


# ---------------------------------------------------------------------------
# Report 2: block recurrence, co-occurrence clusters and dormant filters
# ---------------------------------------------------------------------------
def overlap_stats(block: Iterable[int], draws: Sequence[Sequence[int]]) -> dict:
    """Count how often 5, 4+ and 3+ winning numbers fall inside a block."""
    block_set = set(block)
    overlaps = [len(block_set.intersection(draw)) for draw in draws]
    return {
        "5-Hit": sum(1 for v in overlaps if v == DRAW_SIZE),
        "4-Hit+": sum(1 for v in overlaps if v >= DRAW_SIZE - 1),
        "3-Hit+": sum(1 for v in overlaps if v >= DRAW_SIZE - 2),
        "Average overlap": float(np.mean(overlaps)) if overlaps else 0.0,
    }


def expected_overlap_count(block_size: int, min_overlap: int, n_draws: int) -> float:
    """Expected number of draws with at least ``min_overlap`` numbers inside a block.

    Uses the hypergeometric distribution under a uniform random draw.
    """
    total = math.comb(NUMBER_MAX, DRAW_SIZE)
    favorable = sum(
        math.comb(block_size, t) * math.comb(NUMBER_MAX - block_size, DRAW_SIZE - t)
        for t in range(min_overlap, DRAW_SIZE + 1)
    )
    return n_draws * favorable / total


def sequential_block_report(
    draws: Sequence[Sequence[int]],
    sizes: Sequence[int],
    min_recurrence: int,
) -> pd.DataFrame:
    """Analyse every contiguous 10..14-number block."""
    rows = []
    for size in sizes:
        expected4 = expected_overlap_count(size, DRAW_SIZE - 1, len(draws))
        expected5 = expected_overlap_count(size, DRAW_SIZE, len(draws))
        for start in range(NUMBER_MIN, NUMBER_MAX - size + 2):
            block = tuple(range(start, start + size))
            stats = overlap_stats(block, draws)
            rows.append(
                {
                    "Block type": "Sequential",
                    "Block": "-".join(f"{n:02d}" for n in block),
                    "Size": size,
                    "Start": start,
                    "5-Hit": stats["5-Hit"],
                    "4-Hit+ (recurrence)": stats["4-Hit+"],
                    "3-Hit+": stats["3-Hit+"],
                    "Average overlap": round(stats["Average overlap"], 3),
                    "Expected 4-Hit+": round(expected4, 3),
                    "Expected 5-Hit": round(expected5, 3),
                    "Lift 4-Hit+": round(stats["4-Hit+"] / expected4, 3) if expected4 else 0.0,
                    "Recurrent 3+": "YES" if stats["4-Hit+"] >= min_recurrence else "no",
                    "_numbers": block,
                }
            )
    return pd.DataFrame(rows)


def build_cooccurrence(draws: Sequence[Sequence[int]]) -> Counter:
    """Count how many draws contain each unordered pair of numbers."""
    cooc: Counter = Counter()
    for draw in draws:
        for pair in combinations(sorted(draw), 2):
            cooc[pair] += 1
    return cooc


def grow_cluster(
    seed: Sequence[int],
    size: int,
    cooc: Counter,
    freq: Counter,
) -> tuple[int, ...]:
    """Greedily grow a seed pair into a block of ``size`` numbers.

    At each step, add the number with the strongest total co-occurrence with
    the numbers already in the block. Ties are broken by raw frequency and
    then by number, so the procedure is deterministic.
    """
    block = list(dict.fromkeys(seed))
    universe = list(range(NUMBER_MIN, NUMBER_MAX + 1))

    while len(block) < size:
        remaining = [n for n in universe if n not in block]

        def score(n: int) -> tuple[int, int, int]:
            pair_score = sum(
                cooc.get(tuple(sorted((n, other))), 0) for other in block
            )
            return (pair_score, freq.get(n, 0), -n)

        best = max(remaining, key=score)
        block.append(best)

    return tuple(sorted(block))


def cluster_report(
    draws: Sequence[Sequence[int]],
    size: int,
    n_clusters: int,
    min_new_numbers: int,
) -> pd.DataFrame:
    """Build diverse data-driven blocks from the strongest co-occurring pairs."""
    freq = number_frequencies(draws)
    cooc = build_cooccurrence(draws)

    # Seeds are the most frequent pairs, in descending order.
    seeds = [pair for pair, _ in cooc.most_common()]
    accepted: list[tuple[int, ...]] = []
    rows = []
    expected4 = expected_overlap_count(size, DRAW_SIZE - 1, len(draws))
    expected5 = expected_overlap_count(size, DRAW_SIZE, len(draws))

    for seed in seeds:
        block = grow_cluster(seed, size, cooc, freq)

        # Keep clusters reasonably distinct: at least ``min_new_numbers``
        # numbers must be new with respect to every accepted cluster.
        if any(
            len(set(block).intersection(existing)) > size - min_new_numbers
            for existing in accepted
        ):
            continue

        accepted.append(block)
        stats = overlap_stats(block, draws)
        rows.append(
            {
                "Block type": "Co-occurrence cluster",
                "Block": "-".join(f"{n:02d}" for n in block),
                "Size": size,
                "Seed": f"{seed[0]:02d}-{seed[1]:02d}",
                "5-Hit": stats["5-Hit"],
                "4-Hit+ (recurrence)": stats["4-Hit+"],
                "3-Hit+": stats["3-Hit+"],
                "Average overlap": round(stats["Average overlap"], 3),
                "Expected 4-Hit+": round(expected4, 3),
                "Expected 5-Hit": round(expected5, 3),
                "Lift 4-Hit+": round(stats["4-Hit+"] / expected4, 3) if expected4 else 0.0,
                "Recurrent 3+": "YES" if stats["4-Hit+"] >= MIN_RECURRENCE else "no",
                "_numbers": block,
            }
        )
        if len(rows) >= n_clusters:
            break

    return pd.DataFrame(rows)


def cold_cluster_report(
    draws: Sequence[Sequence[int]],
    size: int,
) -> pd.DataFrame:
    """Analyse blocks made of the least frequent numbers (cold clusters)."""
    freq = number_frequencies(draws)
    ordered = sorted(range(NUMBER_MIN, NUMBER_MAX + 1), key=lambda n: (freq.get(n, 0), n))
    block = tuple(sorted(ordered[:size]))
    stats = overlap_stats(block, draws)
    expected4 = expected_overlap_count(size, DRAW_SIZE - 1, len(draws))
    expected5 = expected_overlap_count(size, DRAW_SIZE, len(draws))

    return pd.DataFrame(
        [
            {
                "Block type": f"Cold cluster ({size})",
                "Block": "-".join(f"{n:02d}" for n in block),
                "Size": size,
                "5-Hit": stats["5-Hit"],
                "4-Hit+ (recurrence)": stats["4-Hit+"],
                "3-Hit+": stats["3-Hit+"],
                "Average overlap": round(stats["Average overlap"], 3),
                "Expected 4-Hit+": round(expected4, 3),
                "Expected 5-Hit": round(expected5, 3),
                "Lift 4-Hit+": round(stats["4-Hit+"] / expected4, 3) if expected4 else 0.0,
                "_numbers": block,
            }
        ]
    )


def number_dormancy_report(window: pd.DataFrame, draws: Sequence[Sequence[int]]) -> pd.DataFrame:
    """Frequency, current gap and last appearance of every number."""
    freq = number_frequencies(draws)
    n_draws = len(draws)
    p = DRAW_SIZE / NUMBER_MAX
    expected = n_draws * p
    sigma = math.sqrt(n_draws * p * (1 - p)) if n_draws else 0.0

    rows = []
    for n in range(NUMBER_MIN, NUMBER_MAX + 1):
        gap = 0
        last_index = None
        for i in range(n_draws - 1, -1, -1):
            if n in draws[i]:
                last_index = i
                break
            gap += 1

        if last_index is None:
            last_contest = None
            last_date = None
        else:
            last_contest = int(window.iloc[last_index]["CONCURSO"])
            last_date = window.iloc[last_index]["FECHA"]

        z_score = (freq.get(n, 0) - expected) / sigma if sigma else 0.0
        rows.append(
            {
                "Number": f"{n:02d}",
                "Frequency": freq.get(n, 0),
                "Expected": round(expected, 2),
                "Z-score": round(z_score, 2),
                "Current gap (draws)": gap,
                "Last contest": last_contest,
                "Last date": f"{last_date:%d/%m/%Y}" if last_date is not None else "-",
                "Low-frequency flag": "YES" if z_score <= -1.5 else "no",
                "Long-gap flag": "YES" if gap >= 10 else "no",
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["Frequency", "Current gap (draws)", "Number"],
        ascending=[True, False, True],
    )


def pair_dormancy_report(draws: Sequence[Sequence[int]]) -> pd.DataFrame:
    """Exact co-occurrence counts for all 378 unordered pairs."""
    freq = number_frequencies(draws)
    cooc = build_cooccurrence(draws)
    n_draws = len(draws)
    p_pair = math.comb(NUMBER_MAX - 2, DRAW_SIZE - 2) / math.comb(NUMBER_MAX, DRAW_SIZE)
    expected = n_draws * p_pair
    p_zero = (1 - p_pair) ** n_draws if n_draws else 1.0

    rows = []
    for pair in combinations(range(NUMBER_MIN, NUMBER_MAX + 1), 2):
        count = cooc.get(pair, 0)
        rows.append(
            {
                "Pair": f"{pair[0]:02d}-{pair[1]:02d}",
                "Occurrences": count,
                "Expected": round(expected, 3),
                "P(0 occurrences)": round(p_zero, 4),
                "Coldness score": freq.get(pair[0], 0) + freq.get(pair[1], 0),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["Occurrences", "Coldness score", "Pair"],
        ascending=[True, True, True],
    )


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def print_table(title: str, df: pd.DataFrame, columns: Sequence[str] | None = None) -> None:
    """Print a compact DataFrame with a title and a separator."""
    print(f"\n{title}")
    print("-" * len(title))
    if df.empty:
        print("(no rows)")
        return

    view = df[list(columns)] if columns else df
    with pd.option_context(
        "display.max_rows", 100,
        "display.max_columns", 20,
        "display.width", 240,
        "display.max_colwidth", 60,
    ):
        print(view.to_string(index=False))


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Save a report table without internal helper columns."""
    public = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")
    public.to_csv(path, index=False)
    print(f"  saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyse the last N Chispazo contests for pool compression and block recurrence.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        default="Chispazo(1).csv",
        help="Path to the Chispazo CSV file.",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=DEFAULT_LAST_CONTESTS,
        help="Number of most recent contests to analyse.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=TOP_POOLS,
        help="Number of pools to show per pool size in Report 1.",
    )
    parser.add_argument(
        "--outdir",
        default="reports",
        help="Directory where CSV report tables are written.",
    )
    args = parser.parse_args()

    if args.last <= 0:
        parser.error("--last must be a positive integer")
    if args.top <= 0:
        parser.error("--top must be a positive integer")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    window, draws = load_dataset(Path(args.csv), args.last)
    freq = number_frequencies(draws)

    # ------------------------------------------------------------------
    # REPORT 1
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("REPORT 1: GAME COMPRESSION / WHEELING POOLS")
    print("=" * 72)
    print(
        "Method: exact enumeration of every pool that can contain at least two\n"
        "complete draws (pair-union candidate generation), followed by exact\n"
        "scoring against the analysis window. A 5-Hit means all five winning\n"
        "numbers are inside the pool. Pools are ranked by 5-Hit, then by the\n"
        "sum of the pool numbers' frequencies (hotness), then numerically."
    )

    for k in POOL_SIZES:
        table = analyze_pools(window, draws, k, args.top)
        expected = table.attrs.get("expected_5hit", float("nan"))
        candidates = table.attrs.get("candidate_count", 0)

        print(
            f"\nPool size {k}: {candidates:,} candidate pools with 2+ five-hits "
            f"were enumerated. Expected 5-Hits for a random {k}-pool in "
            f"{len(draws)} draws = {expected:.3f}."
        )
        print_table(
            f"Top {len(table)} pools of {k} numbers",
            table,
            columns=[
                "K", "Pool size", "Numbers pool", "5-Hit", "4-Hit+",
                "3-Hit+", "Wheel tickets", "Hotness", "Hit contests",
            ],
        )
        save_csv(table, outdir / f"report1_top_pools_{k}.csv")

    # ------------------------------------------------------------------
    # REPORT 2
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("REPORT 2: EXCLUSION PATTERNS AND RECURRENT BLOCKS")
    print("=" * 72)
    print(
        "Definitions used in this report:\n"
        "  * Block recurrence event: a contest where AT LEAST 4 of the 5 drawn\n"
        "    numbers fall inside the block.\n"
        "  * Recurrent block: a block with 3+ recurrence events in the window.\n"
        "  * Lift: observed recurrence count / expected count under a uniform\n"
        "    random draw. Lift > 1 means above-chance recurrence.\n"
        "  * Dormant patterns are descriptive sample properties, not predictions."
    )

    # 2A. Sequential blocks.
    seq = sequential_block_report(draws, BLOCK_SIZES, MIN_RECURRENCE)
    # Rank by lift rather than raw count: larger blocks naturally accumulate
    # more 4+-overlap events, so raw counts are not comparable across sizes.
    recurrent_seq = seq[seq["Recurrent 3+"] == "YES"].sort_values(
        ["Lift 4-Hit+", "4-Hit+ (recurrence)", "5-Hit"],
        ascending=[False, False, False],
    )
    print_table(
        f"2A. Recurrent sequential blocks (top {min(15, len(recurrent_seq))} by lift; "
        f"{len(recurrent_seq)} found)",
        recurrent_seq.head(15),
        columns=[
            "Block", "Size", "5-Hit", "4-Hit+ (recurrence)", "3-Hit+",
            "Expected 4-Hit+", "Lift 4-Hit+", "Recurrent 3+",
        ],
    )
    print(f"  Recurrent sequential blocks found: {len(recurrent_seq)} of {len(seq)} windows.")
    print("  (The complete list is saved in report2_sequential_blocks.csv.)")
    save_csv(seq, outdir / "report2_sequential_blocks.csv")

    # 2B. Data-driven co-occurrence clusters.
    cluster_tables = []
    for size in CLUSTER_SIZES:
        table = cluster_report(draws, size, CLUSTERS_PER_SIZE, MIN_NEW_NUMBERS)
        cluster_tables.append(table)
    clusters = pd.concat(cluster_tables, ignore_index=True) if cluster_tables else pd.DataFrame()
    print_table(
        "2B. Recurrent data-driven blocks (greedy co-occurrence clusters)",
        clusters,
        columns=[
            "Block", "Size", "Seed", "5-Hit", "4-Hit+ (recurrence)",
            "3-Hit+", "Expected 4-Hit+", "Lift 4-Hit+", "Recurrent 3+",
        ],
    )
    save_csv(clusters, outdir / "report2_cooccurrence_clusters.csv")

    # 2C. Cold clusters and dormant blocks.
    cold_tables = [cold_cluster_report(draws, size) for size in CLUSTER_SIZES]
    cold_clusters = pd.concat(cold_tables, ignore_index=True) if cold_tables else pd.DataFrame()
    print_table(
        "2C. Cold clusters (least frequent numbers) - exclusion candidates",
        cold_clusters,
        columns=[
            "Block", "Size", "5-Hit", "4-Hit+ (recurrence)", "3-Hit+",
            "Expected 4-Hit+", "Lift 4-Hit+",
        ],
    )
    save_csv(cold_clusters, outdir / "report2_cold_clusters.csv")

    # Low-recurrence sequential blocks. The threshold is half the random
    # expectation, with a minimum of 0; this identifies under-performing blocks
    # without pretending that "under-performing" is a physical property.
    seq["Dormancy threshold"] = (seq["Expected 4-Hit+"] * 0.5).round(3)
    low_blocks = seq[seq["4-Hit+ (recurrence)"] <= seq["Dormancy threshold"]].sort_values(
        ["4-Hit+ (recurrence)", "Lift 4-Hit+", "Size"],
        ascending=[True, True, False],
    )
    if low_blocks.empty:
        low_blocks = seq.sort_values(
            ["4-Hit+ (recurrence)", "Lift 4-Hit+"],
            ascending=[True, True],
        ).head(10)
        print("\n[note] No sequential block fell below the 0.5x-expectation threshold;"
              " showing the 10 lowest-recurrence windows instead.")
    print_table(
        "2D. Low-recurrence sequential blocks - dormant exclusion candidates",
        low_blocks,
        columns=[
            "Block", "Size", "5-Hit", "4-Hit+ (recurrence)", "3-Hit+",
            "Expected 4-Hit+", "Dormancy threshold", "Lift 4-Hit+",
        ],
    )
    save_csv(low_blocks, outdir / "report2_low_recurrence_blocks.csv")

    # 2E. Individual number dormancy.
    numbers = number_dormancy_report(window, draws)
    print_table(
        "2E. Number frequency and current gap (coldest / longest-gap first)",
        numbers.head(14),
        columns=[
            "Number", "Frequency", "Expected", "Z-score",
            "Current gap (draws)", "Last contest", "Last date",
            "Low-frequency flag", "Long-gap flag",
        ],
    )
    save_csv(numbers, outdir / "report2_number_dormancy.csv")

    # 2F. Pair dormancy.
    pairs = pair_dormancy_report(draws)
    zero_pairs = pairs[pairs["Occurrences"] == 0]
    print_table(
        f"2F. Dormant pairs: {len(zero_pairs)} pairs never appeared together in {len(draws)} draws",
        zero_pairs,
        columns=["Pair", "Occurrences", "Expected", "P(0 occurrences)", "Coldness score"],
    )
    p_pair = math.comb(NUMBER_MAX - 2, DRAW_SIZE - 2) / math.comb(NUMBER_MAX, DRAW_SIZE)
    expected_zero_pairs = math.comb(NUMBER_MAX, 2) * ((1 - p_pair) ** len(draws))
    print(
        f"  Expected number of never-seen pairs under randomness: "
        f"{expected_zero_pairs:.1f}. Observed: {len(zero_pairs)}. "
        "A zero count is therefore not, by itself, evidence of a real bias."
    )
    save_csv(pairs, outdir / "report2_pair_dormancy.csv")

    # 2G. Compact exclusion-filter summary.
    low_freq_numbers = numbers[numbers["Low-frequency flag"] == "YES"]
    long_gap_numbers = numbers[numbers["Long-gap flag"] == "YES"]

    print("\n2G. Exclusion-filter summary (descriptive candidates only)")
    print("-" * 72)
    if low_freq_numbers.empty:
        print("  Low-frequency numbers (Z <= -1.5): none")
    else:
        items = [
            f"{row['Number']} (freq={row['Frequency']}, z={row['Z-score']})"
            for _, row in low_freq_numbers.iterrows()
        ]
        print("  Low-frequency numbers (Z <= -1.5): " + ", ".join(items))
    if long_gap_numbers.empty:
        print("  Long-gap numbers (10+ draws without a hit): none")
    else:
        items = [
            f"{row['Number']} (gap={row['Current gap (draws)']})"
            for _, row in long_gap_numbers.iterrows()
        ]
        print("  Long-gap numbers (10+ draws without a hit): " + ", ".join(items))

    if not low_blocks.empty:
        b = low_blocks.iloc[0]
        print(
            f"  Lowest-recurrence block: {b['Block']} "
            f"({b['4-Hit+ (recurrence)']} occurrences, "
            f"lift={b['Lift 4-Hit+']})"
        )
    if not zero_pairs.empty:
        sample = ", ".join(zero_pairs["Pair"].head(12).tolist())
        print(f"  Never-co-occurring pairs (first 12 of {len(zero_pairs)}): {sample}")

    print("\n  Reminder: random draws have no memory. Backtest any exclusion")
    print("  filter on out-of-sample contests before using it.")

    # ------------------------------------------------------------------
    # Final saved-file index
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("FILES WRITTEN")
    print("=" * 72)
    for path in sorted(outdir.glob("report*.csv")):
        print(f"  {path}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # keep the CLI failure readable
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)

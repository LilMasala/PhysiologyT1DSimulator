"""CLI entry point."""
from __future__ import annotations

import argparse
from pathlib import Path

from t1d_sim import simulate_population


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--outdb")
    p.add_argument("--firebase-project")
    p.add_argument("--outdir", default="./t1d_sim_output/")
    p.add_argument("--n_patients", type=int, default=100)
    p.add_argument("--days", type=int, default=180)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", default="80/10/10")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--male_fraction", type=float, default=0.45)
    p.add_argument("--aid_fraction", type=float, default=0.35,
        help="Fraction of population using AID systems (default 0.35)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not args.outdb and not args.firebase_project:
        raise SystemExit("Either --outdb or --firebase-project is required")
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    simulate_population(
        outdb=args.outdb,
        firebase_project=args.firebase_project,
        outdir=args.outdir,
        n_patients=args.n_patients,
        days=args.days,
        seed=args.seed,
        split=args.split,
        jobs=args.jobs,
        male_fraction=args.male_fraction,
        aid_fraction=args.aid_fraction,
    )


if __name__ == "__main__":
    main()

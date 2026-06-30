"""
=============================================================================
CLI — Build, validate, and query the LoL Draft Model without Streamlit.
=============================================================================

Examples
--------
# Build model and run a quick prediction
python cli.py --data path/to/data.csv predict \
    --team1 "Aatrox,Vi,Ahri,Jinx,Thresh" \
    --team2 "Darius,LeeSin,Azir,Caitlyn,Nautilus"

# Export champion stats and synergy table to JSON
python cli.py --data path/to/data.csv export --out ./stats

# Look up counters
python cli.py --data path/to/data.csv counters --champ Ahri

# Show top synergy pairs
python cli.py --data path/to/data.csv synergies --top 20

# Filter to specific patches
python cli.py --data path/to/data.csv --patches 16.05,16.06 predict \
    --team1 "..." --team2 "..."
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model import DraftModel, ModelConfig, build_model, load_or_build


def parse_team(team_str: str) -> list[str]:
    return [c.strip() for c in team_str.split(",")]


def cmd_predict(model: DraftModel, args):
    team1 = parse_team(args.team1)
    team2 = parse_team(args.team2)

    if len(team1) != 5 or len(team2) != 5:
        print("ERROR: Each team must have exactly 5 champions (comma-separated).")
        sys.exit(1)

    result = model.predict(team1, team2, verbose=True)

    print("\n=== Breakdown ===")
    for k, v in result["breakdown"].items():
        print(f"  {k:35s}: {v:+.4f}")


def cmd_export(model: DraftModel, args):
    model.export_stats(output_dir=args.out)


def cmd_counters(model: DraftModel, args):
    champ = args.champ
    top_n = args.top
    df = model.teamfight_model.best_counters(champ, top_n=top_n)
    print(f"\nTop {top_n} counters to {champ} (by gold advantage at 22 min):")
    print(df.to_string(index=False))


def cmd_synergies(model: DraftModel, args):
    df = model.synergy_model.top_pairs(top_n=args.top)
    print(f"\nTop {args.top} synergy pairs:")
    print(df.to_string(index=False))


def cmd_winrates(model: DraftModel, args):
    df = model.base_model.summary(top_n=args.top)
    print(f"\nTop {args.top} champions by win rate:")
    print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="LoL Draft Betting Model CLI")
    parser.add_argument("--data",    required=True, help="Path to Oracle's Elixir CSV")
    parser.add_argument("--cache",   default="model_cache.pkl", help="Cache file path")
    parser.add_argument("--patches", default="", help="Comma-separated patch list (blank = all)")
    parser.add_argument("--min-champ",  type=int, default=10)
    parser.add_argument("--min-syn",    type=int, default=5)
    parser.add_argument("--min-ctr",    type=int, default=5)
    parser.add_argument("--rebuild",    action="store_true", help="Force rebuild (ignore cache)")

    sub = parser.add_subparsers(dest="command", required=True)

    p_pred = sub.add_parser("predict", help="Predict win probability")
    p_pred.add_argument("--team1", required=True, help="5 champs comma-separated")
    p_pred.add_argument("--team2", required=True, help="5 champs comma-separated")

    p_exp = sub.add_parser("export", help="Export stats to JSON")
    p_exp.add_argument("--out", default="./stats", help="Output directory")

    p_ctr = sub.add_parser("counters", help="Show counters for a champion")
    p_ctr.add_argument("--champ", required=True)
    p_ctr.add_argument("--top", type=int, default=10)

    p_syn = sub.add_parser("synergies", help="Show top synergy pairs")
    p_syn.add_argument("--top", type=int, default=20)

    p_wr = sub.add_parser("winrates", help="Show champion win rates")
    p_wr.add_argument("--top", type=int, default=20)

    args = parser.parse_args()

    # Build config
    config = ModelConfig()
    config.MIN_CHAMP_GAMES   = args.min_champ
    config.MIN_SYNERGY_GAMES = args.min_syn
    config.MIN_COUNTER_GAMES = args.min_ctr
    config.PATCHES = (
        [float(p.strip()) for p in args.patches.split(",") if p.strip()]
        if args.patches.strip() else None
    )
    config.CACHE_PATH = args.cache

    if args.rebuild or not Path(args.cache).exists():
        model = build_model(args.data, config=config, save_cache=True)
    else:
        model = load_or_build(args.data, config=config, cache_path=args.cache)

    dispatch = {
        "predict":   cmd_predict,
        "export":    cmd_export,
        "counters":  cmd_counters,
        "synergies": cmd_synergies,
        "winrates":  cmd_winrates,
    }
    dispatch[args.command](model, args)


if __name__ == "__main__":
    main()

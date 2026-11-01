import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bsgen import botplay, hub
from bsgen.player import PLAY_ACCURACY, play


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="player checkpoint (default: the released model)")
    ap.add_argument("--map", required=True)
    ap.add_argument("--diff", default="ExpertPlus")
    ap.add_argument("--height", type=float, default=1.7, help="player height in metres")
    ap.add_argument("--steps", type=int, default=12,
                    help="sampling steps per two seconds of motion (24 ducks a little deeper under walls)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--warmup", type=float, default=2.0, help="seconds of motion before the first object")
    ap.add_argument("--candidates", type=int, default=4, help="samples per 2 s step; the one that cuts the most notes is kept")
    ap.add_argument("--recenter", type=float, default=0.3, help="share of the stance drift undone per 2 s (0: off)")
    ap.add_argument("--accuracy", type=float, default=PLAY_ACCURACY,
                    help="the accuracy to play like (player models that take one)")
    ap.add_argument("--free-turns", action="store_true",
                    help="don't cap wrist turns at the fastest humans make at that pace")
    ap.add_argument("--out", help="default: next to the map, <map> <diff> (.bsor and .npz)")
    args = ap.parse_args()
    model = botplay.load_player(args.ckpt or hub.player_model(), args.device)
    objects, njs, info = botplay.map_objects(args.map, args.diff, plan=model.config.get("plan", False),
                                             dev=args.device)
    out = Path(args.out or Path(args.map).with_suffix("").as_posix() + f" {args.diff}")
    out = out.with_suffix("") if out.suffix in (".bsor", ".npz") else out
    motion, t0 = play(model, objects, args.height, njs, steps=args.steps, seed=args.seed, dev=args.device,
                      warmup=args.warmup, recenter=args.recenter, candidates=args.candidates,
                      accuracy=args.accuracy, turns=not args.free_turns)
    motion, saved = botplay.touch_up(botplay.from_song_start(motion, t0), objects, njs)
    np.savez(out.with_name(out.name + ".npz"), motion=motion, t0=0.0)
    events = botplay.write_replay(out.with_name(out.name + ".bsor"), args.map, args.diff, motion, info, njs, objects)
    s = botplay.summary(events)
    print(f"{len(motion) / 60:.0f} s of motion: {botplay.describe(s)} ({saved} near misses touched up) -> {out}.bsor")


if __name__ == "__main__":
    main()

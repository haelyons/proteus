"""Predict sharding costs from a recorded single-process stream, without a model."""
from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path


def ratio(a, b):
    return a / b if a is not None and b else None


def candidate_ratio(a, b):
    # CPU's reference backend records no compactions: no increase means ratio 1.
    return 1.0 if a == b == 0 else ratio(a, b)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--batches', required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--ns', type=int, nargs='+', default=[1, 2, 4, 8])
    ap.add_argument('--fixed', required=True, type=json.loads)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    if any(n < 1 for n in args.ns):
        ap.error('--ns must be positive')
    batches = sorted((json.loads(line) for line in Path(args.batches).read_text().splitlines() if line.strip()),
                     key=lambda b: b['batch'])
    run = json.loads(Path(args.run).read_text())
    fixed = args.fixed
    pilot = run.get('seconds', {}).get('pilot', 0)
    # A trailing 25-batch mean, located at the mean n_seen_before of its window.
    curve = []
    for j in range(len(batches)):
        window = batches[max(0, j - 24):j + 1]
        curve.append((sum(b['n_seen_before'] for b in window) / len(window),
                      sum(b['compacted_frac'] for b in window) / len(window)))
    xs = [x for x, _ in curve]

    def fraction(seen):
        if not curve:
            return None
        pos = bisect.bisect_right(xs, seen)
        if pos == 0:
            return curve[0][1]
        if pos == len(curve):
            return curve[-1][1]
        x0, y0 = curve[pos - 1]
        x1, y1 = curve[pos]
        return y0 + (y1 - y0) * (seen - x0) / (x1 - x0)

    def candidates(n):
        seen = [0] * n
        total = 0
        for b in batches:
            i = b['batch'] % n
            total += fraction(seen[i]) * b['n_valid']
            seen[i] += b['n_valid']
        return total

    stream_1 = sum(b['atlas_pass_s'] for b in batches)
    wall_1 = pilot + fixed['load_s'] + stream_1 + fixed['save_s']
    base_candidates = candidates(1)
    records = {}
    for n in args.ns:
        streams = [sum(b['atlas_pass_s'] for b in batches if b['batch'] % n == i) for i in range(n)]
        stream = max(streams)
        wall = (wall_1 if n == 1 else pilot + fixed['load_s'] + stream + fixed['shard_save_s']
                + n * (fixed['merge_load_s'] + fixed['merge_step_s']) + fixed['result_s'] + fixed['save_s'])
        records[str(n)] = {'shard_stream_s': streams, 'stream_pred': stream,
                           'speedup_stream_pred': ratio(stream_1, stream), 'wall_pred': wall,
                           'speedup_e2e_pred': ratio(wall_1, wall),
                           'candidates_ratio_pred': candidate_ratio(candidates(n), base_candidates)}
    Path(args.out).write_text(json.dumps({'n': records, 'candidate_ratio_convention': '0/0 is 1'}, indent=1) + '\n')


if __name__ == '__main__':
    main()

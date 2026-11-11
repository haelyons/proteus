"""Summarize scaling records under OUT; absent measurements remain null."""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import re
import statistics
from pathlib import Path


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else None


def ratio(a, b):
    return a / b if a is not None and b else None


def total(values):
    values = list(values)
    return sum(values) if all(v is not None for v in values) else None


def duration(step):
    if not step or step.get('skipped') or step.get('rc') != 0:
        return None
    return step['end'] - step['start']


def telemetry(rows, batches, gpu):
    keys = ('utilization_mean', 'utilization_p5', 'power_mean', 'memory_used_max')
    empty = dict.fromkeys(keys)
    if not batches or not gpu.get('uuid'):
        return empty
    start = min(b['t_start'] for b in batches if 't_start' in b) if any('t_start' in b for b in batches) else None
    end = max(b['t_end'] for b in batches if 't_end' in b) if any('t_end' in b for b in batches) else None
    if start is None or end is None:
        return empty
    selected = []
    for row in rows:
        if row.get('uuid') != gpu['uuid']:
            continue
        stamp = row.get('timestamp', '')
        try:
            t = float(stamp)
        except ValueError:
            try:
                t = datetime.datetime.strptime(stamp, '%Y/%m/%d %H:%M:%S.%f').timestamp()
            except ValueError:
                continue
        if start <= t <= end:
            selected.append(row)

    def numbers(key):
        result = []
        for row in selected:
            try:
                result.append(float(row[key].split()[0]))
            except (KeyError, ValueError, IndexError):
                pass
        return sorted(result)

    util, power, memory = (numbers(k) for k in ('utilization.gpu', 'power.draw', 'memory.used'))
    # Linear percentile interpolation; nvidia-smi memory.used is MiB.
    p = (len(util) - 1) * .05
    lo = int(p) if util else 0
    return {'utilization_mean': statistics.mean(util) if util else None,
            'utilization_p5': util[lo] + (util[min(lo + 1, len(util) - 1)] - util[lo]) * (p - lo) if util else None,
            'power_mean': statistics.mean(power) if power else None,
            'memory_used_max': max(memory) * 1024**2 if memory else None}


def run_record(path, smi):
    run = read(path / 'run.json', {})
    batches = lines(path / 'batches.jsonl')
    sums = {key: total(b.get(key) for b in batches) if batches is not None else None
            for key in ('atlas_pass_s', 'build_s', 'wait_s', 'seconds')}
    redo = None if batches is None else sum(len(b['redo_layers']) if isinstance(b.get('redo_layers'), list)
                                            else b.get('redo_layers', 0) for b in batches)
    redo_batches = None if batches is None else sum(bool(b.get('redo_layers')) for b in batches)
    model = run.get('model', {})
    layers = model.get('n_layers')
    stats = run.get('stats', {})
    filtered_candidates = stats.get('compacted')
    filtered_layer_batches = (stats['batches'] * layers - stats['dense_layer_batches']
                              if layers is not None and stats.get('batches') is not None
                              and stats.get('dense_layer_batches') is not None else None)
    sizes = read(path / 'output_files.json')
    if sizes is None:
        sizes = {p.name: p.stat().st_size for p in path.glob('*.pt') if p.is_file()}
    return {'seconds': {k: run.get('seconds', {}).get(k) for k in
                        ('load', 'pilot', 'stream_this_process', 'final_save', 'total')},
            'batches': sums, **{k: v for k, v in sums.items() if k != 'seconds'},
            'seconds_sum': sums['seconds'], 'batch_count': len(batches) if batches is not None else None,
            'redo_layers': redo, 'redo_batches': redo_batches,
            'filtered_candidates': filtered_candidates, 'filtered_layer_batches': filtered_layer_batches,
            'stats': {k: run.get('stats', {}).get(k) for k in ('compacted', 'top_cand', 'res_cand')},
            'gpu': {k: run.get('gpu', {}).get(k) for k in ('max_memory_allocated', 'max_memory_reserved', 'uuid')},
            'host': {'peak_rss_bytes': run.get('host', {}).get('peak_rss_bytes')},
            'output_file_bytes': sizes or None, 'smi': telemetry(smi, batches, run.get('gpu', {}))}


def gate(path):
    g = read(path, {})
    return {**{k: g.get(k) for k in ('ok', 'exact_all', 'close_all')},
            'max_rel_diff': {k: v['max_rel_diff'] for k, v in g.get('fields', {}).items() if 'max_rel_diff' in v}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('out', type=Path)
    ap.add_argument('--predictions', type=Path)
    args = ap.parse_args()
    out = args.out
    timeline = read(out / 'timeline.json', {}).get('steps', [])
    steps = {s['step']: s for s in timeline}
    smi = []
    if (out / 'smi.csv').is_file():
        with (out / 'smi.csv').open() as f:
            smi = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(f)]
    single = run_record(out / 'single', smi)
    pilot_s = duration(steps.get('pilot'))
    wall_1 = total([pilot_s, duration(steps.get('single'))])
    ns = {1, 2, 4, 8}
    ns.update(int(p.name[1:]) for p in out.glob('n[0-9]*') if p.name[1:].isdigit())
    ns.update(int(m.group(1)) for s in steps if (m := re.fullmatch(r'shards_n(\d+)', s)))
    predictions = read(args.predictions, {}) if args.predictions else {}
    records = {}
    for n in sorted(ns):
        root = out / f'n{n}'
        runs = [single] if n == 1 else [run_record(root / f's{i}', smi) for i in range(n)]
        streams = [r['seconds']['stream_this_process'] for r in runs]
        complete = all(s is not None for s in streams)
        stream = max(streams) if complete else None
        wall = wall_1 if n == 1 else total([pilot_s, duration(steps.get(f'shards_n{n}')),
                                                        duration(steps.get(f'merge_n{n}'))])
        merge = read(root / 'merged' / 'merge.json', {})
        merge_record = {k: merge.get(k) for k in ('validate_s', 'steps', 'result_s', 'save_s', 'total_s', 'peak_rss_bytes')}
        time_path = root / 'merge_time.txt'
        match = re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)', time_path.read_text()) if time_path.is_file() else None
        merge_record['time_v_peak_rss_bytes'] = int(match.group(1)) * 1024 if match else None
        record = {'pilot_s': pilot_s, 'wall_1': wall_1, 'wall_N': wall, 'speedup_e2e': ratio(wall_1, wall),
                  'efficiency': ratio(ratio(wall_1, wall), n), 'stream': stream,
                  'speedup_stream': ratio(single['seconds']['stream_this_process'], stream),
                  'imbalance': ratio(stream, statistics.mean(streams)) if complete else None,
                  'runs': runs, 'merge': merge_record if n > 1 else None,
                  'gate': gate(root / 'gate_compare.json') if n > 1 else None,
                  'vs_ref': {'ok': read(out / 'single_vs_ref.json' if n == 1 else root / 'vs_ref.json', {}).get('ok')},
                  'compare_s': duration(steps.get(f'compare_n{n}')),
                  'vs_ref_s': duration(steps.get('single_vs_ref' if n == 1 else f'vs_ref_n{n}'))}
        for key in ('compacted', 'top_cand', 'res_cand'):
            name = 'candidates_ratio' if key == 'compacted' else key + '_ratio'
            record[name] = ratio(total(r['stats'][key] for r in runs), single['stats'][key])
        record['redo_layers_ratio'] = ratio(total(r['redo_layers'] for r in runs), single['redo_layers'])
        filtered_mean = ratio(total(r['filtered_candidates'] for r in runs),
                              total(r['filtered_layer_batches'] for r in runs))
        single_mean = ratio(single['filtered_candidates'], single['filtered_layer_batches'])
        record['candidates_per_filtered_layer_batch_ratio'] = ratio(filtered_mean, single_mean)
        pred = predictions.get('n', {}).get(str(n), {})
        if args.predictions:
            record['predictions'] = {}
            for key, measured in (('stream_pred', stream), ('wall_pred', wall),
                                  ('speedup_stream_pred', record['speedup_stream']),
                                  ('speedup_e2e_pred', record['speedup_e2e']),
                                  ('candidates_ratio_pred', record['candidates_ratio']),
                                  ('redo_layer_batches_pred', total(r['redo_layers'] for r in runs))):
                value = pred.get(key)
                record['predictions'][key] = {'predicted': value, 'measured': measured,
                                               'measured_over_predicted': ratio(measured, value)}
            pred_streams = pred.get('shard_stream_s')
            record['predictions']['shard_stream_s'] = {
                'predicted': pred_streams, 'measured': streams,
                'measured_over_predicted': [ratio(s, p) for s, p in zip(streams, pred_streams)]
                if pred_streams is not None and len(pred_streams) == len(streams) else None}
        records[str(n)] = record
    naive = {'cmp_prefix': read(out / 'naive' / 'cmp_prefix.json', {'ok': None})}
    for pair in ('prefix', 'steady'):
        filtered = run_record(out / 'naive' / pair / 'filtered', smi)
        plain = run_record(out / 'naive' / pair / 'naive', smi)
        a = lines(out / 'naive' / pair / 'filtered' / 'batches.jsonl')
        b = lines(out / 'naive' / pair / 'naive' / 'batches.jsonl')
        same = a is not None and b is not None and [r['batch'] for r in a] == [r['batch'] for r in b]
        naive[pair] = {'filtered': filtered, 'naive': plain, 'same_batches': same,
                       **{key + '_ratio': ratio(plain['batches'][key], filtered['batches'][key]) if same else None
                          for key in ('seconds', 'atlas_pass_s')}}
    report = {'pilot_s': pilot_s, 'wall_1': wall_1, 'single': single, 'n': records, 'naive': naive,
              'cross_gpu': read(out / 'digest_compare.json'), 'steps': timeline}
    (out / 'scaling.json').write_text(json.dumps(report, indent=1) + '\n')
    md = ['# Scaling measurements', '', '## N versus 1', '',
          '| N (`n.N`) | wall_N (s) | speedup_e2e | efficiency | speedup_stream | candidates_ratio | imbalance |',
          '| --- | --- | --- | --- | --- | --- | --- |']
    for n, r in records.items():
        md.append('| ' + ' | '.join([n] + [str(r[k]) for k in ('wall_N', 'speedup_e2e', 'efficiency',
                                                             'speedup_stream', 'candidates_ratio', 'imbalance')]) + ' |')
    md += ['', '## filtered versus naive', '',
           '| pair (`naive.pair`) | seconds_ratio | atlas_pass_s_ratio |', '| --- | --- | --- |']
    for pair in ('prefix', 'steady'):
        md.append(f"| {pair} | {naive[pair]['seconds_ratio']} | {naive[pair]['atlas_pass_s_ratio']} |")
    md += ['', f"Prefix gate (`naive.cmp_prefix.ok`): {naive['cmp_prefix'].get('ok')}", '']
    for n, r in records.items():
        md.append(f"N={n}: `n.{n}.gate` = {json.dumps(r['gate'])}; `n.{n}.vs_ref.ok` = {r['vs_ref']['ok']}")
    md += ['', f"Cross-GPU (`cross_gpu`): {json.dumps(report['cross_gpu'])}", '']
    if args.predictions:
        md += ['Predicted/measured comparisons: `n.N.predictions` in scaling.json.', '']
    (out / 'scaling.md').write_text('\n'.join(md))


if __name__ == '__main__':
    main()

"""Recorded unit_writes cases vs (A) the legacy unit_writes.py hook and (B) atlas.intervene, both run here (CPU fp32, 1 thread)."""
import json, sys, torch
sys.path.insert(0, '/home/hal/dev/proteus')
from atlas.intervene import Intervener, Edit
torch.set_num_threads(1); torch.set_grad_enabled(False)
seqs = json.load(open(sys.argv[1])); rows = [json.loads(l) for l in open('/home/hal/dev/proteus/results/unit_writes_8M_20261105/cases.jsonl')]
M = Intervener(); model, layers, tok = M.model, M.layers, M.tok
edit = {}
def legacy_hook(l):
    def hk(_m, _i, o):
        if edit.get('l') == l:
            o = o.clone(); o[0, edit['p'] + 1, edit['u']] = 0.0
        return o
    return hk
def legacy_lp(x, l=None, u=None, p=None):
    hs = [lay.intermediate.register_forward_hook(legacy_hook(i)) for i, lay in enumerate(layers)]
    edit.update(l=l, u=u, p=p)
    try:
        return torch.log_softmax(model(input_ids=x).logits[0].double(), -1)
    finally:
        edit.clear(); [h.remove() for h in hs]
def eff(lp0, lp1, x, p):
    t = int(x[0, p + 1]); kl = (lp0.exp() * (lp0 - lp1)).sum(-1).clamp(min=0)
    other = torch.cat([kl[1:p + 1], kl[p + 2:-1]])
    return {'dlogp_true': float(lp1[p + 1, t] - lp0[p + 1, t]), 'kl': float(kl[p + 1]), 'travel_kl': float(other.max()) if other.numel() else 0.0}
keys = ['dlogp_true', 'kl', 'travel_kl']
stats = {'legacy_vs_rec': [], 'module_vs_legacy_bitwise': 0, 'module_vs_rec': [], 'n': 0, 'seq_len_mismatch': 0}
cache = {}
for r in rows:
    s = seqs[r['acc']][:1022]
    if r['pos'] >= len(s): stats['seq_len_mismatch'] += 1; continue
    x = tok(s, return_tensors='pt')['input_ids']
    if r['acc'] not in cache: cache[r['acc']] = legacy_lp(x)
    lp0 = cache[r['acc']]
    for unit, prefix in ((r['unit'], ''), (r['base_unit'], 'base_')):
        A = eff(lp0, legacy_lp(x, r['layer'], unit, r['pos']), x, r['pos'])
        r_mod = M.run(x, Edit(r['layer'], torch.nn.functional.one_hot(torch.tensor(unit), 1280).float(), [r['pos']], [0.0]))
        B = eff(lp0, r_mod.lp_full, x, r['pos'])
        stats['module_vs_legacy_bitwise'] += all(A[k] == B[k] for k in keys)
        stats['legacy_vs_rec'].append(max(abs(A[k] - r[prefix + k]) for k in keys))
        stats['module_vs_rec'].append(max(abs(B[k] - r[prefix + k]) for k in keys))
        stats['n'] += 1
L, Mr = torch.tensor(stats['legacy_vs_rec']), torch.tensor(stats['module_vs_rec'])
print(json.dumps({'effects_compared': stats['n'], 'seq_len_mismatch_cases': stats['seq_len_mismatch'],
  'module_equals_legacy_bitwise': stats['module_vs_legacy_bitwise'],
  'legacy_vs_recorded': {'bitwise': int((L == 0).sum()), 'max_abs': float(L.max()), 'median_abs': float(L.median())},
  'module_vs_recorded': {'bitwise': int((Mr == 0).sum()), 'max_abs': float(Mr.max()), 'median_abs': float(Mr.median())},
  'torch': torch.__version__}, indent=1))

#!/bin/bash
# AlphaFold DB descriptors (ss3, contacts, burial, pLDDT, phi/psi, C-beta coordinates) for every
# accession of the Swiss-Prot 2026_03 build, held-out set included, from the v6 Swiss-Prot archive.
# The archive is fetched over NPART parallel ranged requests (one connection ran at ~3.5 MB/s on
# 2026-11-04). Then: whole-build summary (status counts, pLDDT bands, held-out subset) and a sha256
# manifest of every run file on this filesystem (to check committed copies against originals).
# Env: JOB OUT FS DEADLINE. SMOKE=1: local range-serving URL (SMOKE_URL), accession list SMOKE_ACCS.
set -uo pipefail
SMOKE=${SMOKE:-0}
HELD=heldout_created_after_2021-11-17.tsv
if [ "$SMOKE" = 1 ]; then
  URL=$SMOKE_URL; DATA=data/swissprot_2026_03.pt; NPROC=2; NPART=4; ACC="--acc-file $SMOKE_ACCS"; AF=$OUT/afdb; HELD=lists/$HELD
else
  URL=https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/swissprot_pdb_v6.tar
  DATA=$FS/cache/swissprot_2026_03.pt; NPROC=$(( $(nproc) - 2 )); NPART=24; ACC=""; AF=$HOME/afdb
  python3 -m pip install --user -q pydssp 2>&1 | tail -1
fi
left() { echo $(( DEADLINE - $(date +%s) )); }
echo "[atlas] start $(date -u +%FT%TZ) job=$JOB smoke=$SMOKE ($(left)s left); $(nproc) cpus, disk free $(df -BG --output=avail ${HOME} | tail -1)"
mkdir -p $AF/parts $AF/pdb
SIZE=$(curl -sfI "$URL" | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}' | tail -1)
echo "[atlas] archive $URL: $SIZE bytes"
[ -n "$SIZE" ] || { echo "[atlas] no size from HEAD"; exit 3; }
CH=$(( (SIZE + NPART - 1) / NPART ))
t0=$(date +%s)
for i in $(seq 0 $(( NPART - 1 ))); do
  s=$(( i * CH )); e=$(( s + CH - 1 )); [ $e -ge $SIZE ] && e=$(( SIZE - 1 ))
  f=$AF/parts/$(printf %04d $i)
  ( for try in 1 2 3 4 5 6 7 8; do
      curl -sf -r $s-$e -o $f "$URL" && [ "$(stat -c %s $f)" = "$(( e - s + 1 ))" ] && exit 0
      sleep $(( try * 15 ))
    done; echo "[atlas] part $i failed" ) &
done
wait
TOT=$(stat -c %s $AF/parts/* | awk '{s += $1} END {print s}')
DT=$(( $(date +%s) - t0 ))
echo "[atlas] downloaded $TOT of $SIZE bytes in ${DT}s ($(( TOT / (DT + 1) / 1000000 )) MB/s)"
[ "$TOT" = "$SIZE" ] || { echo "[atlas] download incomplete: stopping"; exit 4; }
t1=$(date +%s)
cat $(ls $AF/parts/* | sort) | tar -x -C $AF/pdb --wildcards '*-F1-model_v*.pdb.gz'
rm -rf $AF/parts
echo "[atlas] extracted $(find $AF/pdb -name '*.pdb.gz' | wc -l) F1 models in $(( $(date +%s) - t1 ))s"
timeout $(( $(left) - 900 )) python3 scripts/afdb_descriptors.py --data $DATA --pdb-dir $AF/pdb --procs $NPROC $ACC \
  --out $FS/descriptors_swissprot_2026_03.pt 2>&1 | tail -2 | sed 's/^/[atlas] descriptors: /'
python3 - <<PY 2>&1 | sed 's/^/[atlas] /'
import torch, json, collections
d = torch.load("$FS/descriptors_swissprot_2026_03.pt", weights_only=False)
st = collections.Counter(d["status"].values()); n = sum(st.values())
held = {ln.split("\t")[0] for ln in open("$HELD") if ln.strip() and not ln.startswith("Entry")}
p = d["plddt"].float(); ss = d["ss3"]; off = d["offset"]
hm = torch.zeros(p.numel(), dtype=torch.bool)
for i, a in enumerate(d["acc"]):
    if a in held: hm[off[i]:off[i + 1]] = True
def bands(m):
    q = p[m]
    return {"lt50": round(float((q < 50).float().mean()), 4), "50to70": round(float(((q >= 50) & (q < 70)).float().mean()), 4),
            "lt70": round(float((q < 70).float().mean()), 4), "ge90": round(float((q >= 90).float().mean()), 4), "residues": int(m.sum())}
out = {"accessions": n, "status": dict(st), "frac_status": {k: round(v / n, 5) for k, v in st.items()},
       "plddt_all": bands(torch.ones_like(hm)), "plddt_heldout": bands(hm), "plddt_train": bands(~hm),
       "ss3_frac_plddt_ge70": [round(float((ss[p >= 70] == k).float().mean()), 4) for k in range(3)],
       "heldout_ok": sum(1 for a, s in d["status"].items() if a in held and s == "ok"), "heldout_listed": len(held),
       "mismatch_accessions": sorted(a for a, s in d["status"].items() if s == "sequence-mismatch")[:2000]}
json.dump(out, open("$OUT/descriptor_summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "mismatch_accessions"}))
PY
( cd $FS/runs 2>/dev/null && find . -type f ! -name '*.pt' ! -name '*.gz' -size -200M -print0 | sort -z | xargs -0 sha256sum ) > $OUT/fs_manifest.sha256 2>/dev/null
echo "[atlas] manifest: $(wc -l < $OUT/fs_manifest.sha256) files hashed under $FS/runs"
echo "[atlas] end $(date -u +%FT%TZ)"

#!/bin/bash
# Measurement #5: strict held-out set. Search the 10,010 date-held-out Swiss-Prot sequences against the
# representative sequences of UniRef50 release 2021_04 (ESM-2's training clustering) with MMseqs2, and classify
# each query by its best hit under stated identity / coverage criteria (scripts/strict_heldout.py).
# The release exists only as one 169 GB archive (uniref2021_04.tar.gz: nested uniref100.tar, uniref90.tar,
# uniref50.tar); it is fetched over NPART ranged requests to local disk, its MD5 checked against the release
# metalink while it is decompressed, and only the uniref50 member is kept (its FASTA copied to the filesystem).
# Env: JOB OUT FS DEADLINE; SMOKE=1: SMOKE_URL (file:// archive), SMOKE_MD5, SMOKE_DESC_ACCS, a few hundred queries.
set -uo pipefail
SMOKE=${SMOKE:-0}
HELD=heldout_created_after_2021-11-17.tsv
SENS=5.7; EVAL=1e-3; MAXSEQS=1000          # MMseqs2 search settings (recorded in settings.json)
if [ "$SMOKE" = 1 ]; then
  URL=$SMOKE_URL; MD5=$SMOKE_MD5; DATA=data/swissprot_2026_03.pt; HELD=lists/$HELD; NPART=4; W=$OUT/work; TH=2; MAXQ=300
  KEEP=$OUT/keep; DESC_ACCS=$SMOKE_DESC_ACCS
else
  URL=https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2021_04/uniref/uniref2021_04.tar.gz
  MD5=444f0a7062a65a988ba1d5949d1d6419       # RELEASE.metalink, release 2021_04
  DATA=$FS/cache/swissprot_2026_03.pt; NPART=16; W=$HOME/strict_work; TH=$(nproc); MAXQ=0
  KEEP=$FS/cache/uniref50_2021_04; DESC_ACCS=$OUT/descriptor_accs.txt
  sudo apt-get install -y -q mmseqs2 pigz > $OUT/apt.log 2>&1 || echo "[atlas] apt install failed (see apt.log)"
fi
left() { echo $(( DEADLINE - $(date +%s) )); }
command -v pigz > /dev/null && UNZ="pigz -dc" || UNZ="gzip -dc"
mkdir -p $W/parts $KEEP
echo "[atlas] start $(date -u +%FT%TZ) job=$JOB smoke=$SMOKE ($(left)s left); $(nproc) cpus; local disk free $(df -BG --output=avail $W | tail -1); $(mmseqs version 2>/dev/null | head -1 | sed 's/^/mmseqs /')"

FASTA=$KEEP/uniref50.fasta.gz
if [ -s $FASTA ] && [ -s $KEEP/archive.md5 ]; then
  echo "[atlas] reusing $FASTA (archive md5 $(cat $KEEP/archive.md5))"
else
  SIZE=$(curl -sfI "$URL" | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}' | tail -1)
  [ -n "$SIZE" ] || { echo "[atlas] no size from HEAD"; exit 3; }
  echo "[atlas] archive $URL: $SIZE bytes, $NPART ranges"
  CH=$(( (SIZE + NPART - 1) / NPART )); t0=$(date +%s)
  for i in $(seq 0 $(( NPART - 1 ))); do
    s=$(( i * CH )); e=$(( s + CH - 1 )); [ $e -ge $SIZE ] && e=$(( SIZE - 1 ))
    f=$W/parts/$(printf %04d $i)
    ( for try in 1 2 3 4 5 6 7 8; do
        curl -sf -r $s-$e -o $f "$URL" && [ "$(stat -c %s $f)" = "$(( e - s + 1 ))" ] && exit 0
        sleep $(( try * 15 ))
      done; echo "[atlas] part $i failed" ) &
  done
  wait
  TOT=$(stat -c %s $W/parts/* | awk '{s += $1} END {print s}'); DT=$(( $(date +%s) - t0 ))
  echo "[atlas] downloaded $TOT of $SIZE bytes in ${DT}s ($(( TOT / (DT + 1) / 1000000 )) MB/s)"
  [ "$TOT" = "$SIZE" ] || { echo "[atlas] download incomplete: stopping"; exit 4; }
  t1=$(date +%s)
  # one pass: MD5 of the archive, decompression, and extraction of the uniref50 member only
  cat $(ls $W/parts/* | sort) | tee >(md5sum | cut -d' ' -f1 > $W/archive.md5) | $UNZ | tar -xf - -C $W --wildcards '*uniref50*'
  for i in $(seq 120); do [ -s $W/archive.md5 ] && break; sleep 1; done; GOT=$(cat $W/archive.md5)
  echo "[atlas] decompressed and extracted in $(( $(date +%s) - t1 ))s; md5 $GOT (expected $MD5)"
  [ "$GOT" = "$MD5" ] || { echo "[atlas] MD5 MISMATCH: stopping"; exit 5; }
  rm -rf $W/parts
  ls -la $W | sed 's/^/[atlas]   /'
  NESTED=$(find $W -maxdepth 2 -name 'uniref50*.tar' | head -1)
  if [ -n "$NESTED" ]; then
    tar -tvf $NESTED > $OUT/uniref50_member_listing.txt
    tar -xf $NESTED -C $W --wildcards '*fasta.gz' '*release_note' 2>/dev/null
  fi
  F=$(find $W -name 'uniref50*fasta.gz' | head -1)
  [ -n "$F" ] || { echo "[atlas] no uniref50 FASTA in the archive (listing in uniref50_member_listing.txt): stopping"; exit 6; }
  cp $F $FASTA.tmp && mv $FASTA.tmp $FASTA
  find $W -name '*release_note' -exec cp {} $KEEP/ \;
  echo $GOT > $KEEP/archive.md5
fi
cp $KEEP/archive.md5 $OUT/ 2>/dev/null; cat $KEEP/*release_note 2>/dev/null | sed 's/^/[atlas]   /' | head -8
NSEQ=$($UNZ $FASTA | grep -c '^>')
echo "[atlas] UniRef50 2021_04 FASTA: $NSEQ sequences ($(stat -c %s $FASTA) bytes)"

python3 scripts/strict_heldout.py queries --data $DATA --held $HELD --out $W/queries.fasta --max-queries $MAXQ | sed 's/^/[atlas] /'
if [ "$SMOKE" != 1 ] && [ -f $FS/descriptors_swissprot_2026_03.pt ]; then
  python3 -c "
import torch; d = torch.load('$FS/descriptors_swissprot_2026_03.pt', weights_only=False)
open('$DESC_ACCS', 'w').write('\n'.join(d['acc']) + '\n'); print('[atlas] descriptor accessions', len(d['acc']))"
fi
t2=$(date +%s)
mmseqs createdb $FASTA $W/tdb > $OUT/mmseqs_createdb.log 2>&1 && mmseqs createdb $W/queries.fasta $W/qdb >> $OUT/mmseqs_createdb.log 2>&1 \
  || { echo "[atlas] createdb failed"; tail -5 $OUT/mmseqs_createdb.log; exit 7; }
echo "[atlas] databases built in $(( $(date +%s) - t2 ))s"
t3=$(date +%s)
timeout $(( $(left) - 300 )) mmseqs search $W/qdb $W/tdb $W/res $W/tmp -s $SENS -e $EVAL --max-seqs $MAXSEQS --alignment-mode 3 \
  --cov-mode 0 -c 0 --threads $TH > $OUT/mmseqs_search.log 2>&1 || { echo "[atlas] search failed or timed out"; tail -5 $OUT/mmseqs_search.log; exit 8; }
mmseqs convertalis $W/qdb $W/tdb $W/res $W/hits.m8 --format-output "query,target,fident,alnlen,qcov,tcov,evalue,bits,qlen,tlen" --threads $TH > /dev/null 2>&1
echo "[atlas] search $(( $(date +%s) - t3 ))s; $(wc -l < $W/hits.m8) hits"
python3 - <<PY
import json; json.dump({"reference": "UniRef50 release 2021_04 (27-Oct-2021), representative sequences, from $URL",
  "archive_md5": open("$KEEP/archive.md5").read().strip(), "sequences": $NSEQ,
  "mmseqs": "$(mmseqs version 2>/dev/null | head -1)", "search": {"sensitivity_s": $SENS, "evalue": "$EVAL", "max_seqs": $MAXSEQS,
  "alignment_mode": 3, "coverage_filter_in_search": "none (cov-mode 0, c 0); criteria applied afterwards"},
  "fields": "query,target,fident,alnlen,qcov,tcov,evalue,bits,qlen,tlen"}, open("$OUT/settings.json", "w"), indent=1)
PY
DA=""; [ -s "$DESC_ACCS" ] && DA="--desc-accs $DESC_ACCS"
python3 scripts/strict_heldout.py classify --hits $W/hits.m8 --queries $W/queries.fasta --out $OUT $DA | sed 's/^/[atlas] /' | cut -c1-900
gzip -c $W/hits.m8 > $OUT/hits.m8.gz
echo "[atlas] end $(date -u +%FT%TZ)"

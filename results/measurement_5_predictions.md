# Measurement #5: strict held-out set (written 2026-11-05, before the job runs)

Job `jobs/strict_heldout` (smoke-tested on CPU with a nested fake archive: 50/50 planted relatives at ~85%
identity caught by the 50% criterion with the right target, 49/50 at ~40% caught only by the 30% criterion).

**Reference.** UniRef50 release 2021_04 (27 Oct 2021; 49,874,565 clusters, per its release note), the
release ESM-2's training set was clustered with. It is only available as `uniref2021_04.tar.gz`
(169,188,630,923 bytes, MD5 444f0a7062a65a988ba1d5949d1d6419 from RELEASE.metalink), whose first member is
`uniref100.tar`; the job downloads it over 16 ranged requests, checks the MD5 during decompression and
keeps the UniRef50 FASTA (copied to `esm-atlas-results/cache/uniref50_2021_04/`).
**What the targets are:** cluster representative sequences only (one per cluster), not members.
**Queries:** the 10,010 Swiss-Prot 2026_03 entries created after 2021-11-17, full length.
**Search:** MMseqs2 (Ubuntu 22.04 package; version recorded), `search -s 5.7 -e 1e-3 --max-seqs 1000
--alignment-mode 3`, no coverage filter in the search; criteria applied afterwards:

| criterion | meaning |
|---|---|
| id50_tcov80 | a hit >= 50% identical over >= 80% of the representative (UniRef50's own membership rule, applied to the representative) |
| id50_qcov80 | >= 50% over >= 80% of the query |
| id40_cov50, id30_cov50 | >= 40% / >= 30% identity over >= 50% of either sequence |
| any_hit | any hit at E <= 1e-3 |

**What a negative result establishes.** No representative within the criterion. Members of a UniRef50 cluster
are >= 50% identical to its seed, which is not always the representative shown, and ESM-2 sampled UniRef90
members inside clusters; so "no hit >= 50% to any representative" does not exclude a >= 50%-identical
training member. The 30% criterion and "no hit at all" are the stricter proxies; none is a proof of
absence from training.

| quantity | predicted |
|---|---|
| archive MD5 | matches |
| download (16 ranges) | 6-20 min (89 MB/s on one connection from this container) |
| FASTA sequences | 49,874,565 |
| search | 20-60 min on 30 cores |
| survivors, id50_tcov80 | 5-20% of 10,010 |
| survivors, id30_cov50 | 1-5% |
| survivors, no hit at all | 0.5-3% |
| survivors that also have descriptors (AFDB model; 7,208 of the 10,010 do) | 40-70% of each survivor set (the newest, least-studied entries are both likelier to be novel and likelier to lack a model) |

## Addendum after the first attempt (11:27 UTC), before the second

The first attempt downloaded the archive (169,188,630,923 bytes in 729 s, 231 MB/s) and its MD5 matched; then it
stopped by design: `uniref50.tar` holds `README`, `uniref50.release_note`, `uniref50.xml.gz` (25,107,210,941
bytes) and `uniref.xsd`, with no FASTA. The extracted tar was on the instance's local disk and was lost at
termination (my design: it should have been kept on the filesystem before the check). The second attempt
streams archive -> nested tar -> XML -> representative FASTA in one pass straight onto the filesystem
(`jobs/strict_heldout/xml2fasta.awk`; smoke-tested on a nested fake in UniProt's XML layout). The converter also
counts how many representatives are their cluster's seed (`isSeed` on the representative only), which says how
often "representative" and "seed" (the sequence UniRef50 membership is defined against) coincide. Predicted:
49,874,565 entries and sequences; representatives that are seeds: 30-70%. Cap 2 h (first attempt ~$0.55).

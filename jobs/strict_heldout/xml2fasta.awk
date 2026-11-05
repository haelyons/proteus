# UniRef XML (one element per line, as UniProt writes it) -> FASTA of the representative sequences.
# Header: >UniRef50_ID seed=1|0 (whether the representative is the cluster's seed). Counts to stderr.
/<entry id=/ { match($0, /id="[^"]+"/); id = substr($0, RSTART + 4, RLENGTH - 5); inrep = 0; seed = "0"; ne++; next }
/<representativeMember>/ { inrep = 1; next }
/<\/representativeMember>/ { inrep = 0; next }
/isSeed/ { if (inrep && $0 ~ /"true"/) seed = "1"; next }
/<sequence / { if (inrep) { s = $0; sub(/.*<sequence[^>]*>/, "", s); sub(/<\/sequence>.*/, "", s); print ">" id " seed=" seed; print s; n++; if (seed == "1") ns++ } }
END { print "entries " ne " sequences " n " representative_is_seed " ns > "/dev/stderr" }

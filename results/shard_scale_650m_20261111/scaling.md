# Scaling measurements

## N versus 1

| N (`n.N`) | wall_N (s) | speedup_e2e | efficiency | speedup_stream | candidates_ratio | imbalance |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 3851.598347902298 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 2 | 2124.03213429451 | 1.8133427859751252 | 0.9066713929875626 | 1.9504749367955945 | 1.6646509916356063 | 1.0008443801756286 |
| 4 | 1256.1153898239136 | 3.066277492581496 | 0.766569373145374 | 3.7037199233226374 | 2.3279039907728856 | 1.0075529205247455 |
| 8 | 838.3458030223846 | 4.594283568924191 | 0.5742854461155239 | 6.906349466854182 | 2.644102354964941 | 1.0105340614711074 |

## filtered versus naive

| pair (`naive.pair`) | seconds_ratio | atlas_pass_s_ratio |
| --- | --- | --- |
| prefix | 1.997830448558011 | 1.9979246832632935 |
| steady | 1.2696166779025748 | 1.2696695513319323 |

Prefix gate (`naive.cmp_prefix.ok`): True

N=1: `n.1.gate` = null; `n.1.vs_ref.ok` = True
N=2: `n.2.gate` = {"ok": true, "exact_all": true, "close_all": true, "max_rel_diff": {"sub_sum": 0.0, "sub_sq": 4.675481264122051e-17, "gram": 2.870973995122312e-07}}; `n.2.vs_ref.ok` = True
N=4: `n.4.gate` = {"ok": true, "exact_all": true, "close_all": true, "max_rel_diff": {"sub_sum": 0.0, "sub_sq": 4.675481264122051e-17, "gram": 4.04760268164785e-07}}; `n.4.vs_ref.ok` = True
N=8: `n.8.gate` = {"ok": true, "exact_all": true, "close_all": true, "max_rel_diff": {"sub_sum": 0.0, "sub_sq": 4.675481264122051e-17, "gram": 2.557206345382169e-07}}; `n.8.vs_ref.ok` = True

Cross-GPU (`cross_gpu`): {"all_equal": true, "pairs": [{"gpu": 1, "equal": true, "first_diff": null}, {"gpu": 2, "equal": true, "first_diff": null}, {"gpu": 3, "equal": true, "first_diff": null}, {"gpu": 4, "equal": true, "first_diff": null}, {"gpu": 5, "equal": true, "first_diff": null}, {"gpu": 6, "equal": true, "first_diff": null}, {"gpu": 7, "equal": true, "first_diff": null}]}

# 8M comparison

Calibration passes: True
H1: against

| Comparison | Cliff's d | p | Bootstrap CI |
| --- | --- | --- | --- |
| N vs S1 | 0.23643279121376404 | 1.7429858091031726e-28 | [0.19577062119899608, 0.27649643093059467] |
| N vs S2 | 0.21419210853292747 | 1.0171560864159371e-23 | [0.17366816305078006, 0.2569688934283707] |
| N vs S3 | 0.1978389348767724 | 2.2075579045795456e-20 | [0.1568208617951875, 0.2374315506112055] |
| N vs R | -0.020525580540462607 | 0.3304902373109538 | [-0.061837355657607646, 0.023367257068421493] |
| S1 vs RS1 | -0.046967552157840875 | 0.02789086517703551 | [-0.09043110973472687, -0.0033279364310087193] |
| S2 vs RS2 | 0.010608235963839308 | 0.6190724822661523 | [-0.03474861108134874, 0.05325083363069808] |
| S3 vs RS3 | -0.027818083788155423 | 0.1940222402011952 | [-0.06793412253237358, 0.01490529138565432] |

Mean d(N, Sc): 0.21615461154115465
H1 conditions: {"attributable": false, "d_ge_0147": false, "half_gap": false, "p_lt_001": false}

| Source | n | Residue modes (k/n; share; CI) | Median primary JSD | Median peak log2 | Pooled v1.2 calls |
| --- | --- | --- | --- | --- | --- |
| N | 1499 | 1453/1499; 0.9693128752501667; [0.9592781940767103, 0.9774474176209083] | 0.18920424580574036 | 1.2046091556549072 | {"composition association": 20, "context association": 1397, "insufficient samples": 1, "no association found by this procedure": 2, "structural association": 80} |
| R | 1499 | 1490/1499; 0.9939959973315544; [0.9886332557681567, 0.9972510216601171] | 0.18934570252895355 | 1.191031813621521 | {"composition association": 2, "context association": 1485, "insufficient samples": 1, "structural association": 12} |
| S1 | 1427 | 1427/1427; 1.0; [0.9974182792052455, 1.0] | 0.13960451632738113 | 1.555954098701477 | {"composition association": 25, "context association": 1401, "insufficient samples": 73, "no association found by this procedure": 1} |
| S2 | 1433 | 1432/1433; 0.9993021632937893; [0.9961180990822305, 0.9999823324603405] | 0.14307338744401932 | 1.602064609527588 | {"composition association": 24, "context association": 1405, "insufficient samples": 67, "no association found by this procedure": 1, "structural association": 3} |
| S3 | 1420 | 1420/1420; 1.0; [0.9974055689362493, 1.0] | 0.14823263883590698 | 1.6289527416229248 | {"composition association": 23, "context association": 1395, "insufficient samples": 80, "structural association": 2} |
| RS1 | 1498 | 1498/1498; 1.0; [0.9975404932041227, 1.0] | 0.14930333942174911 | 1.5726689100265503 | {"composition association": 33, "context association": 1464, "insufficient samples": 2, "structural association": 1} |
| RS2 | 1498 | 1497/1498; 0.9993324432576769; [0.996286282399907, 0.9999830990694016] | 0.14264319837093353 | 1.6053857207298279 | {"composition association": 23, "context association": 1472, "insufficient samples": 2, "no association found by this procedure": 1, "structural association": 2} |
| RS3 | 1489 | 1489/1489; 1.0; [0.9975256455526764, 1.0] | 0.1516546532511711 | 1.6521968841552734 | {"composition association": 14, "context association": 1472, "insufficient samples": 11, "structural association": 3} |

Bootstrap: {"draws": 2000, "level": 0.95, "seed": 0}
Reproduction: {"N": {"identical": 1500, "n": 1500}, "S1": {"identical": 1500, "n": 1500}, "S2": {"identical": 1500, "n": 1500}, "S3": {"identical": 1500, "n": 1500}}

# Ruler checks

## Planted columns

### randdir

| Gate | Pass/total |
| --- | --- |
| buried | 12/12 |
| helix | 12/12 |
| hydro_win | 12/12 |
| lys_helix | 12/12 |
| narrow | 3/12 |
| noise + every alpha = 0 | 537/542 |
| ordered | 12/12 |
| trp | 12/12 |
| two_job | 12/12 |

Noise null: 6/518; rate 0.011583011583011582; 95% Clopper-Pearson [0.004262297187012031, 0.02503987414150373].

| Type@alpha | Calls | Structural k/n (share) | Median primary JSD | Residue modes |
| --- | --- | --- | --- | --- |
| buried@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.016775730066001415 | {"H": 1} |
| buried@0.5 | {"structural association": 3} | 3/3 (1) | 0.33432501554489136 | {} |
| buried@1.0 | {"structural association": 3} | 3/3 (1) | 0.37441299855709076 | {} |
| buried@2.0 | {"structural association": 3} | 3/3 (1) | 0.38119545578956604 | {"P": 1} |
| buried@3.0 | {"structural association": 3} | 3/3 (1) | 0.380568191409111 | {} |
| buried@4.0 | {"structural association": 3} | 3/3 (1) | 0.3947299122810364 | {} |
| helix@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.01781107671558857 | {} |
| helix@0.5 | {"structural association": 3} | 3/3 (1) | 0.3621682822704315 | {"E": 1} |
| helix@1.0 | {"structural association": 3} | 3/3 (1) | 0.3852359801530838 | {} |
| helix@2.0 | {"structural association": 3} | 3/3 (1) | 0.3989375978708267 | {} |
| helix@3.0 | {"structural association": 3} | 3/3 (1) | 0.3987070322036743 | {"Q": 1} |
| helix@4.0 | {"structural association": 3} | 3/3 (1) | 0.3909485936164856 | {} |
| hydro_win@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.012627221643924713 | {} |
| hydro_win@0.5 | {"no association found by this procedure": 1, "structural association": 2} | 3/3 (1) | 0.0700082890689373 | {"L": 2, "V": 2, "I": 1} |
| hydro_win@1.0 | {"structural association": 1, "context association": 2} | 3/3 (1) | 0.15029113739728928 | {"I": 3, "L": 2, "V": 3} |
| hydro_win@2.0 | {"context association": 3} | 3/3 (1) | 0.2620583325624466 | {"I": 3, "L": 3, "V": 3} |
| hydro_win@3.0 | {"context association": 3} | 3/3 (1) | 0.2554994449019432 | {"I": 3, "L": 3, "V": 3} |
| hydro_win@4.0 | {"context association": 3} | 3/3 (1) | 0.2656572237610817 | {"I": 3, "L": 3, "V": 3} |
| lys_helix@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.011440258473157883 | {} |
| lys_helix@0.5 | {"composition association": 3} | 1/3 (0.333333) | 0.035337142646312714 | {"K": 3} |
| lys_helix@1.0 | {"context association": 3} | 3/3 (1) | 0.3423765003681183 | {"K": 3} |
| lys_helix@2.0 | {"context association": 3} | 3/3 (1) | 0.38526657223701477 | {"K": 3} |
| lys_helix@3.0 | {"context association": 3} | 3/3 (1) | 0.38526657223701477 | {"K": 3} |
| lys_helix@4.0 | {"context association": 3} | 3/3 (1) | 0.38526657223701477 | {"K": 3} |
| mix_diff_20@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.007319711148738861 | {"E": 1} |
| mix_diff_20@0.5 | {"structural association": 3} | 3/3 (1) | 0.0857660248875618 | {"H": 1} |
| mix_diff_20@1.0 | {"structural association": 3} | 3/3 (1) | 0.09435293450951576 | {"R": 1} |
| mix_diff_20@2.0 | {"structural association": 3} | 3/3 (1) | 0.08883947134017944 | {} |
| mix_diff_20@3.0 | {"structural association": 2, "context association": 1} | 3/3 (1) | 0.09218868985772133 | {"D": 2, "R": 2, "Q": 1} |
| mix_diff_20@4.0 | {"structural association": 2, "context association": 1} | 3/3 (1) | 0.0826034676283598 | {"R": 1, "Q": 1} |
| mix_diff_2@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.007975596934556961 | {} |
| mix_diff_2@0.5 | {"composition association": 3} | 0/3 (0) | 0.02576306089758873 | {"E": 2, "K": 3} |
| mix_diff_2@1.0 | {"context association": 3} | 3/3 (1) | 0.2694706320762634 | {"E": 3, "K": 3} |
| mix_diff_2@2.0 | {"context association": 3} | 3/3 (1) | 0.2705889195203781 | {"E": 3, "K": 3} |
| mix_diff_2@3.0 | {"context association": 3} | 3/3 (1) | 0.27067457139492035 | {"E": 3, "K": 3} |
| mix_diff_2@4.0 | {"context association": 3} | 3/3 (1) | 0.2705889195203781 | {"E": 3, "K": 3} |
| mix_diff_8@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.015645131468772888 | {} |
| mix_diff_8@0.5 | {"context association": 1, "no association found by this procedure": 1, "composition association": 1} | 1/3 (0.333333) | 0.052545879036188126 | {"K": 1, "Q": 2, "R": 2} |
| mix_diff_8@1.0 | {"context association": 3} | 3/3 (1) | 0.09816285595297813 | {"K": 2, "Q": 3, "D": 2, "A": 1, "R": 1} |
| mix_diff_8@2.0 | {"context association": 3} | 3/3 (1) | 0.09725641831755638 | {"K": 2, "Q": 3, "R": 3, "A": 1, "D": 1} |
| mix_diff_8@3.0 | {"context association": 3} | 3/3 (1) | 0.09719700552523136 | {"D": 2, "K": 2, "Q": 3, "R": 2} |
| mix_diff_8@4.0 | {"context association": 2, "structural association": 1} | 3/3 (1) | 0.10286307334899902 | {"K": 3, "R": 2, "A": 1, "D": 1, "Q": 1} |
| mix_same_20@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.013321258127689362 | {} |
| mix_same_20@0.5 | {"structural association": 3} | 3/3 (1) | 0.2627980038523674 | {"G": 1} |
| mix_same_20@1.0 | {"structural association": 3} | 3/3 (1) | 0.3878716975450516 | {} |
| mix_same_20@2.0 | {"structural association": 3} | 3/3 (1) | 0.38700658082962036 | {} |
| mix_same_20@3.0 | {"structural association": 3} | 3/3 (1) | 0.38091522455215454 | {} |
| mix_same_20@4.0 | {"structural association": 3} | 3/3 (1) | 0.3923987150192261 | {} |
| mix_same_2@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.020030049607157707 | {"I": 1} |
| mix_same_2@0.5 | {"context association": 3} | 3/3 (1) | 0.0871342085301876 | {"E": 3, "K": 3} |
| mix_same_2@1.0 | {"context association": 3} | 3/3 (1) | 0.311001218855381 | {"E": 3, "K": 3} |
| mix_same_2@2.0 | {"context association": 3} | 3/3 (1) | 0.3548339009284973 | {"E": 3, "K": 3} |
| mix_same_2@3.0 | {"context association": 3} | 3/3 (1) | 0.35692715644836426 | {"E": 3, "K": 3} |
| mix_same_2@4.0 | {"context association": 3} | 3/3 (1) | 0.3512447476387024 | {"E": 3, "K": 3} |
| mix_same_8@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.016385572962462902 | {} |
| mix_same_8@0.5 | {"structural association": 3} | 3/3 (1) | 0.2899952009320259 | {"A": 1, "E": 1} |
| mix_same_8@1.0 | {"structural association": 3} | 3/3 (1) | 0.35564136505126953 | {"E": 1, "K": 1, "Q": 2, "S": 1} |
| mix_same_8@2.0 | {"structural association": 3} | 3/3 (1) | 0.3596384897828102 | {"K": 1, "E": 1, "S": 1, "A": 1} |
| mix_same_8@3.0 | {"structural association": 3} | 3/3 (1) | 0.35282251238822937 | {"A": 1, "E": 1} |
| mix_same_8@4.0 | {"structural association": 3} | 3/3 (1) | 0.35792218148708344 | {"Q": 2, "K": 1} |
| narrow@0.0 | {"no association found by this procedure": 3} | 1/3 (0.333333) | 0.012705907225608826 | {} |
| narrow@0.5 | {"composition association": 3} | 1/3 (0.333333) | 0.02750600315630436 | {"S": 3, "T": 3} |
| narrow@1.0 | {"context association": 3} | 3/3 (1) | 0.27527087926864624 | {"S": 3, "T": 3} |
| narrow@2.0 | {"context association": 3} | 3/3 (1) | 0.4927242398262024 | {"S": 3, "T": 3} |
| narrow@3.0 | {"context association": 3} | 3/3 (1) | 0.4926556348800659 | {"S": 3, "T": 3} |
| narrow@4.0 | {"context association": 3} | 3/3 (1) | 0.49318817257881165 | {"S": 3, "T": 3} |
| noise@0.0 | {"no association found by this procedure": 498, "structural association": 2, "context association": 1, "composition association": 2} | 6/503 (0.0119284) | 0.012641898356378078 | {"M": 6, "D": 1, "A": 5, "V": 2, "R": 3, "G": 4, "W": 3, "L": 1, "I": 2, "E": 4, "Y": 2, "C": 2, "P": 1, "S": 3, "Q": 4, "N": 3, "T": 1, "H": 1} |
| noise@0.5 | {"no association found by this procedure": 3} | 0/3 (0) | 0.007539466023445129 | {"E": 1} |
| noise@1.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.008503429591655731 | {} |
| noise@2.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.014649668708443642 | {"G": 1} |
| noise@3.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.017494551837444305 | {} |
| noise@4.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.023578678257763386 | {"T": 1} |
| ordered@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.020301327109336853 | {} |
| ordered@0.5 | {"structural association": 3} | 3/3 (1) | 0.23436281085014343 | {} |
| ordered@1.0 | {"structural association": 3} | 3/3 (1) | 0.2639341354370117 | {} |
| ordered@2.0 | {"structural association": 3} | 3/3 (1) | 0.24644093960523605 | {} |
| ordered@3.0 | {"structural association": 3} | 3/3 (1) | 0.25519968569278717 | {} |
| ordered@4.0 | {"structural association": 3} | 3/3 (1) | 0.25833000987768173 | {} |
| part_diff_0@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.009393486194312572 | {} |
| part_diff_0@0.5 | {"composition association": 3} | 0/3 (0) | 0.039230963215231895 | {"K": 3} |
| part_diff_0@1.0 | {"context association": 3} | 3/3 (1) | 0.29521822929382324 | {"K": 3} |
| part_diff_0@2.0 | {"context association": 3} | 3/3 (1) | 0.38526657223701477 | {"K": 3} |
| part_diff_0@3.0 | {"context association": 3} | 3/3 (1) | 0.38526657223701477 | {"K": 3} |
| part_diff_0@4.0 | {"context association": 3} | 3/3 (1) | 0.38526657223701477 | {"K": 3} |
| part_diff_10@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.010479927062988281 | {"P": 1} |
| part_diff_10@0.5 | {"composition association": 1, "context association": 2} | 3/3 (1) | 0.09965547919273376 | {"V": 3} |
| part_diff_10@1.0 | {"context association": 3} | 3/3 (1) | 0.30205005407333374 | {"V": 3} |
| part_diff_10@2.0 | {"context association": 3} | 3/3 (1) | 0.46284249424934387 | {"V": 3} |
| part_diff_10@3.0 | {"context association": 3} | 3/3 (1) | 0.4628424197435379 | {"V": 3} |
| part_diff_10@4.0 | {"context association": 3} | 3/3 (1) | 0.46284249424934387 | {"V": 3} |
| part_diff_11@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.011618118733167648 | {"F": 1} |
| part_diff_11@0.5 | {"composition association": 1, "no association found by this procedure": 2} | 0/3 (0) | 0.017607556656003 | {"I": 3} |
| part_diff_11@1.0 | {"context association": 3} | 3/3 (1) | 0.37256741523742676 | {"I": 3} |
| part_diff_11@2.0 | {"context association": 3} | 3/3 (1) | 0.6225050836801529 | {"I": 3} |
| part_diff_11@3.0 | {"context association": 3} | 3/3 (1) | 0.6225050836801529 | {"I": 3} |
| part_diff_11@4.0 | {"context association": 3} | 3/3 (1) | 0.6225050836801529 | {"I": 3} |
| part_diff_12@0.0 | {"no association found by this procedure": 3} | 1/3 (0.333333) | 0.008859120309352875 | {"K": 1, "R": 1} |
| part_diff_12@0.5 | {"context association": 1, "composition association": 1, "no association found by this procedure": 1} | 1/3 (0.333333) | 0.03940551541745663 | {"G": 3} |
| part_diff_12@1.0 | {"context association": 3} | 3/3 (1) | 0.36658936738967896 | {"G": 3} |
| part_diff_12@2.0 | {"context association": 3} | 3/3 (1) | 0.6001782417297363 | {"G": 3} |
| part_diff_12@3.0 | {"context association": 3} | 3/3 (1) | 0.6001782417297363 | {"G": 3} |
| part_diff_12@4.0 | {"context association": 3} | 3/3 (1) | 0.6001782417297363 | {"G": 3} |
| part_diff_13@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.00681507564149797 | {} |
| part_diff_13@0.5 | {"composition association": 2, "context association": 1} | 2/3 (0.666667) | 0.04986722208559513 | {"F": 3} |
| part_diff_13@1.0 | {"context association": 3} | 3/3 (1) | 0.2519585117697716 | {"F": 3} |
| part_diff_13@2.0 | {"context association": 3} | 3/3 (1) | 0.5717944651842117 | {"F": 3} |
| part_diff_13@3.0 | {"context association": 3} | 3/3 (1) | 0.5717944651842117 | {"F": 3} |
| part_diff_13@4.0 | {"context association": 3} | 3/3 (1) | 0.5717944651842117 | {"F": 3} |
| part_diff_14@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.0087728351354599 | {} |
| part_diff_14@0.5 | {"composition association": 2, "no association found by this procedure": 1} | 0/3 (0) | 0.028647745959460735 | {"Y": 3} |
| part_diff_14@1.0 | {"context association": 3} | 3/3 (1) | 0.3168337494134903 | {"Y": 3} |
| part_diff_14@2.0 | {"context association": 3} | 3/3 (1) | 0.5446189641952515 | {"Y": 3} |
| part_diff_14@3.0 | {"context association": 3} | 3/3 (1) | 0.5446189641952515 | {"Y": 3} |
| part_diff_14@4.0 | {"context association": 3} | 3/3 (1) | 0.5446189641952515 | {"Y": 3} |
| part_diff_15@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.036435963585972786 | {} |
| part_diff_15@0.5 | {"composition association": 3} | 0/3 (0) | 0.021725581027567387 | {"H": 3, "Q": 1} |
| part_diff_15@1.0 | {"context association": 3} | 3/3 (1) | 0.21053268760442734 | {"H": 3} |
| part_diff_15@2.0 | {"context association": 3} | 3/3 (1) | 0.45259687304496765 | {"H": 3} |
| part_diff_15@3.0 | {"context association": 3} | 3/3 (1) | 0.45259687304496765 | {"H": 3} |
| part_diff_15@4.0 | {"context association": 3} | 3/3 (1) | 0.4525968134403229 | {"H": 3} |
| part_diff_16@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.007835057564079762 | {"E": 1} |
| part_diff_16@0.5 | {"no association found by this procedure": 3} | 0/3 (0) | 0.005597962997853756 | {} |
| part_diff_16@1.0 | {"context association": 3} | 3/3 (1) | 0.25496330857276917 | {"M": 3} |
| part_diff_16@2.0 | {"context association": 3} | 3/3 (1) | 0.6780321896076202 | {"M": 3} |
| part_diff_16@3.0 | {"context association": 3} | 3/3 (1) | 0.6780321896076202 | {"M": 3} |
| part_diff_16@4.0 | {"context association": 3} | 3/3 (1) | 0.6780321896076202 | {"M": 3} |
| part_diff_17@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.018906113225966692 | {"H": 1, "K": 1} |
| part_diff_17@0.5 | {"no association found by this procedure": 2, "composition association": 1} | 0/3 (0) | 0.020357667468488216 | {"W": 1} |
| part_diff_17@1.0 | {"context association": 3} | 3/3 (1) | 0.09218479692935944 | {"W": 3} |
| part_diff_17@2.0 | {"context association": 3} | 3/3 (1) | 0.5320122241973877 | {"W": 3} |
| part_diff_17@3.0 | {"context association": 3} | 3/3 (1) | 0.5702565014362335 | {"W": 3} |
| part_diff_17@4.0 | {"context association": 3} | 3/3 (1) | 0.5702564716339111 | {"W": 3} |
| part_diff_18@0.0 | {"no association found by this procedure": 3} | 1/3 (0.333333) | 0.01213621161878109 | {} |
| part_diff_18@0.5 | {"composition association": 1, "no association found by this procedure": 2} | 0/3 (0) | 0.008587223012000322 | {"C": 2} |
| part_diff_18@1.0 | {"context association": 3} | 3/3 (1) | 0.17468144744634628 | {"C": 3} |
| part_diff_18@2.0 | {"context association": 3} | 3/3 (1) | 0.4757852256298065 | {"C": 3} |
| part_diff_18@3.0 | {"context association": 3} | 3/3 (1) | 0.4757852256298065 | {"C": 3} |
| part_diff_18@4.0 | {"context association": 3} | 3/3 (1) | 0.4757852256298065 | {"C": 3} |
| part_diff_19@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.010669517796486616 | {"G": 1} |
| part_diff_19@0.5 | {"no association found by this procedure": 3} | 0/3 (0) | 0.020800543949007988 | {"T": 1} |
| part_diff_19@1.0 | {"context association": 3} | 3/3 (1) | 0.33982735872268677 | {"P": 3} |
| part_diff_19@2.0 | {"context association": 3} | 3/3 (1) | 0.7920278012752533 | {"P": 3} |
| part_diff_19@3.0 | {"context association": 3} | 3/3 (1) | 0.7920278012752533 | {"P": 3} |
| part_diff_19@4.0 | {"context association": 3} | 3/3 (1) | 0.7920278012752533 | {"P": 3} |
| part_diff_1@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.005075568333268166 | {} |
| part_diff_1@0.5 | {"context association": 1, "composition association": 2} | 2/3 (0.666667) | 0.044622790068387985 | {"E": 3} |
| part_diff_1@1.0 | {"context association": 3} | 3/3 (1) | 0.487398236989975 | {"E": 3} |
| part_diff_1@2.0 | {"context association": 3} | 3/3 (1) | 0.7306002974510193 | {"E": 3} |
| part_diff_1@3.0 | {"context association": 3} | 3/3 (1) | 0.7306002974510193 | {"E": 3} |
| part_diff_1@4.0 | {"context association": 3} | 3/3 (1) | 0.7306002974510193 | {"E": 3} |
| part_diff_2@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.019558286294341087 | {"K": 1} |
| part_diff_2@0.5 | {"composition association": 2, "context association": 1} | 1/3 (0.333333) | 0.04994147829711437 | {"D": 3} |
| part_diff_2@1.0 | {"context association": 3} | 3/3 (1) | 0.2898466885089874 | {"D": 3} |
| part_diff_2@2.0 | {"context association": 3} | 3/3 (1) | 0.37018750607967377 | {"D": 3} |
| part_diff_2@3.0 | {"context association": 3} | 3/3 (1) | 0.3701874911785126 | {"D": 3} |
| part_diff_2@4.0 | {"context association": 3} | 3/3 (1) | 0.37018750607967377 | {"D": 3} |
| part_diff_3@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.021050382405519485 | {} |
| part_diff_3@0.5 | {"composition association": 2, "context association": 1} | 2/3 (0.666667) | 0.04954253323376179 | {"R": 3} |
| part_diff_3@1.0 | {"context association": 3} | 3/3 (1) | 0.28821374475955963 | {"R": 3} |
| part_diff_3@2.0 | {"context association": 3} | 3/3 (1) | 0.36848604679107666 | {"R": 3} |
| part_diff_3@3.0 | {"context association": 3} | 3/3 (1) | 0.36848604679107666 | {"R": 3} |
| part_diff_3@4.0 | {"context association": 3} | 3/3 (1) | 0.36848604679107666 | {"R": 3} |
| part_diff_4@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.009014040697366 | {} |
| part_diff_4@0.5 | {"composition association": 2, "context association": 1} | 1/3 (0.333333) | 0.03973880410194397 | {"L": 3} |
| part_diff_4@1.0 | {"context association": 3} | 3/3 (1) | 0.5001891851425171 | {"L": 3} |
| part_diff_4@2.0 | {"context association": 3} | 3/3 (1) | 0.640703946352005 | {"L": 3} |
| part_diff_4@3.0 | {"context association": 3} | 3/3 (1) | 0.640703946352005 | {"L": 3} |
| part_diff_4@4.0 | {"context association": 3} | 3/3 (1) | 0.640703946352005 | {"L": 3} |
| part_diff_5@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.019220557995140553 | {} |
| part_diff_5@0.5 | {"composition association": 2, "context association": 1} | 1/3 (0.333333) | 0.027842652052640915 | {"A": 3} |
| part_diff_5@1.0 | {"context association": 3} | 3/3 (1) | 0.4127778112888336 | {"A": 3} |
| part_diff_5@2.0 | {"context association": 3} | 3/3 (1) | 0.5838223546743393 | {"A": 3} |
| part_diff_5@3.0 | {"context association": 3} | 3/3 (1) | 0.5838223546743393 | {"A": 3} |
| part_diff_5@4.0 | {"context association": 3} | 3/3 (1) | 0.5838223546743393 | {"A": 3} |
| part_diff_6@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.027491718530654907 | {} |
| part_diff_6@0.5 | {"composition association": 3} | 0/3 (0) | 0.009719846304506063 | {"Q": 3} |
| part_diff_6@1.0 | {"context association": 3} | 3/3 (1) | 0.24471262842416763 | {"Q": 3} |
| part_diff_6@2.0 | {"context association": 3} | 3/3 (1) | 0.35205410420894623 | {"Q": 3} |
| part_diff_6@3.0 | {"context association": 3} | 3/3 (1) | 0.35205410420894623 | {"Q": 3} |
| part_diff_6@4.0 | {"context association": 3} | 3/3 (1) | 0.35205410420894623 | {"Q": 3} |
| part_diff_7@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.020910300314426422 | {} |
| part_diff_7@0.5 | {"no association found by this procedure": 2, "context association": 1} | 1/3 (0.333333) | 0.015719033777713776 | {"P": 1, "S": 2} |
| part_diff_7@1.0 | {"context association": 3} | 3/3 (1) | 0.4241318851709366 | {"S": 3} |
| part_diff_7@2.0 | {"context association": 3} | 3/3 (1) | 0.7237382233142853 | {"S": 3} |
| part_diff_7@3.0 | {"context association": 3} | 3/3 (1) | 0.7237382233142853 | {"S": 3} |
| part_diff_7@4.0 | {"context association": 3} | 3/3 (1) | 0.7237382233142853 | {"S": 3} |
| part_diff_8@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.00772363506257534 | {"N": 1} |
| part_diff_8@0.5 | {"context association": 1, "composition association": 2} | 1/3 (0.333333) | 0.023103754967451096 | {"T": 3, "S": 1} |
| part_diff_8@1.0 | {"context association": 3} | 3/3 (1) | 0.3453257083892822 | {"T": 3} |
| part_diff_8@2.0 | {"context association": 3} | 3/3 (1) | 0.4714662879705429 | {"T": 3} |
| part_diff_8@3.0 | {"context association": 3} | 3/3 (1) | 0.4714662879705429 | {"T": 3} |
| part_diff_8@4.0 | {"context association": 3} | 3/3 (1) | 0.4714662879705429 | {"T": 3} |
| part_diff_9@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.015405058860778809 | {"D": 1, "Y": 1} |
| part_diff_9@0.5 | {"no association found by this procedure": 1, "composition association": 1, "context association": 1} | 1/3 (0.333333) | 0.030874809250235558 | {"N": 3} |
| part_diff_9@1.0 | {"context association": 3} | 3/3 (1) | 0.3936111629009247 | {"N": 3} |
| part_diff_9@2.0 | {"context association": 3} | 3/3 (1) | 0.5058882385492325 | {"N": 3} |
| part_diff_9@3.0 | {"context association": 3} | 3/3 (1) | 0.5058882385492325 | {"N": 3} |
| part_diff_9@4.0 | {"context association": 3} | 3/3 (1) | 0.5058882385492325 | {"N": 3} |
| part_same_0@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.010314002633094788 | {"A": 1} |
| part_same_0@0.5 | {"context association": 3} | 3/3 (1) | 0.06389042548835278 | {"K": 3} |
| part_same_0@1.0 | {"context association": 3} | 3/3 (1) | 0.31334012746810913 | {"K": 3} |
| part_same_0@2.0 | {"context association": 3} | 3/3 (1) | 0.38526657223701477 | {"K": 3} |
| part_same_0@3.0 | {"context association": 3} | 3/3 (1) | 0.38526657223701477 | {"K": 3} |
| part_same_0@4.0 | {"context association": 3} | 3/3 (1) | 0.38526657223701477 | {"K": 3} |
| part_same_10@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.009733682498335838 | {} |
| part_same_10@0.5 | {"context association": 3} | 3/3 (1) | 0.05834432505071163 | {"V": 3} |
| part_same_10@1.0 | {"context association": 3} | 3/3 (1) | 0.3939278721809387 | {"V": 3} |
| part_same_10@2.0 | {"context association": 3} | 3/3 (1) | 0.4317833483219147 | {"V": 3} |
| part_same_10@3.0 | {"context association": 3} | 3/3 (1) | 0.4317833483219147 | {"V": 3} |
| part_same_10@4.0 | {"context association": 3} | 3/3 (1) | 0.4317833483219147 | {"V": 3} |
| part_same_11@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.006798945367336273 | {"P": 1} |
| part_same_11@0.5 | {"composition association": 2, "context association": 1} | 1/3 (0.333333) | 0.028094127774238586 | {"I": 3} |
| part_same_11@1.0 | {"context association": 3} | 3/3 (1) | 0.29332560300827026 | {"I": 3} |
| part_same_11@2.0 | {"context association": 3} | 3/3 (1) | 0.3694564551115036 | {"I": 3} |
| part_same_11@3.0 | {"context association": 3} | 3/3 (1) | 0.3694564551115036 | {"I": 3} |
| part_same_11@4.0 | {"context association": 3} | 3/3 (1) | 0.3694564551115036 | {"I": 3} |
| part_same_12@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.010225832462310791 | {"M": 1} |
| part_same_12@0.5 | {"context association": 3} | 3/3 (1) | 0.06088462099432945 | {"G": 3} |
| part_same_12@1.0 | {"context association": 3} | 3/3 (1) | 0.4342108517885208 | {"G": 3} |
| part_same_12@2.0 | {"context association": 3} | 3/3 (1) | 0.6001782417297363 | {"G": 3} |
| part_same_12@3.0 | {"context association": 3} | 3/3 (1) | 0.6001782417297363 | {"G": 3} |
| part_same_12@4.0 | {"context association": 3} | 3/3 (1) | 0.6001782417297363 | {"G": 3} |
| part_same_13@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.03292485699057579 | {} |
| part_same_13@0.5 | {"composition association": 3} | 0/3 (0) | 0.011261637322604656 | {"F": 3} |
| part_same_13@1.0 | {"context association": 3} | 3/3 (1) | 0.28071337938308716 | {"F": 3} |
| part_same_13@2.0 | {"context association": 3} | 3/3 (1) | 0.37702496349811554 | {"F": 3} |
| part_same_13@3.0 | {"context association": 3} | 3/3 (1) | 0.37702496349811554 | {"F": 3} |
| part_same_13@4.0 | {"context association": 3} | 3/3 (1) | 0.37702496349811554 | {"F": 3} |
| part_same_14@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.004979630932211876 | {} |
| part_same_14@0.5 | {"composition association": 3} | 0/3 (0) | 0.022721909452229738 | {"Y": 3, "D": 1} |
| part_same_14@1.0 | {"context association": 3} | 3/3 (1) | 0.3087385892868042 | {"Y": 3} |
| part_same_14@2.0 | {"context association": 3} | 3/3 (1) | 0.39137980341911316 | {"Y": 3} |
| part_same_14@3.0 | {"context association": 3} | 3/3 (1) | 0.39137980341911316 | {"Y": 3} |
| part_same_14@4.0 | {"context association": 3} | 3/3 (1) | 0.39137980341911316 | {"Y": 3} |
| part_same_15@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.015249287709593773 | {} |
| part_same_15@0.5 | {"composition association": 3} | 0/3 (0) | 0.014468122273683548 | {"H": 3} |
| part_same_15@1.0 | {"context association": 3} | 3/3 (1) | 0.2214880809187889 | {"H": 3} |
| part_same_15@2.0 | {"context association": 3} | 3/3 (1) | 0.45259687304496765 | {"H": 3} |
| part_same_15@3.0 | {"context association": 3} | 3/3 (1) | 0.45259687304496765 | {"H": 3} |
| part_same_15@4.0 | {"context association": 3} | 3/3 (1) | 0.45259687304496765 | {"H": 3} |
| part_same_16@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.003541461075656116 | {} |
| part_same_16@0.5 | {"composition association": 2, "no association found by this procedure": 1} | 0/3 (0) | 0.01435808977112174 | {"M": 2} |
| part_same_16@1.0 | {"context association": 3} | 3/3 (1) | 0.27307218313217163 | {"M": 3} |
| part_same_16@2.0 | {"context association": 3} | 3/3 (1) | 0.3695078194141388 | {"M": 3} |
| part_same_16@3.0 | {"context association": 3} | 3/3 (1) | 0.3695078194141388 | {"M": 3} |
| part_same_16@4.0 | {"context association": 3} | 3/3 (1) | 0.3695078194141388 | {"M": 3} |
| part_same_17@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.015899034217000008 | {} |
| part_same_17@0.5 | {"composition association": 2, "context association": 1} | 1/3 (0.333333) | 0.020402023568749428 | {"W": 3} |
| part_same_17@1.0 | {"context association": 3} | 3/3 (1) | 0.16635113954544067 | {"W": 3} |
| part_same_17@2.0 | {"context association": 3} | 3/3 (1) | 0.33689049631357193 | {"W": 3} |
| part_same_17@3.0 | {"context association": 3} | 3/3 (1) | 0.33689049631357193 | {"W": 3} |
| part_same_17@4.0 | {"context association": 3} | 3/3 (1) | 0.33689049631357193 | {"W": 3} |
| part_same_18@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.009189341217279434 | {} |
| part_same_18@0.5 | {"no association found by this procedure": 3} | 0/3 (0) | 0.01117064617574215 | {"C": 1} |
| part_same_18@1.0 | {"context association": 3} | 3/3 (1) | 0.2513826936483383 | {"C": 3} |
| part_same_18@2.0 | {"context association": 3} | 3/3 (1) | 0.4757852256298065 | {"C": 3} |
| part_same_18@3.0 | {"context association": 3} | 3/3 (1) | 0.4757852256298065 | {"C": 3} |
| part_same_18@4.0 | {"context association": 3} | 3/3 (1) | 0.4757852256298065 | {"C": 3} |
| part_same_19@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.008670177776366472 | {} |
| part_same_19@0.5 | {"composition association": 3} | 0/3 (0) | 0.025869159027934074 | {"P": 3} |
| part_same_19@1.0 | {"context association": 3} | 3/3 (1) | 0.30145810544490814 | {"P": 3} |
| part_same_19@2.0 | {"context association": 3} | 3/3 (1) | 0.6315755248069763 | {"P": 3} |
| part_same_19@3.0 | {"context association": 3} | 3/3 (1) | 0.6315754950046539 | {"P": 3} |
| part_same_19@4.0 | {"context association": 3} | 3/3 (1) | 0.6315755248069763 | {"P": 3} |
| part_same_1@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.007269883994013071 | {} |
| part_same_1@0.5 | {"context association": 1, "composition association": 2} | 2/3 (0.666667) | 0.054103925824165344 | {"E": 3} |
| part_same_1@1.0 | {"context association": 3} | 3/3 (1) | 0.2757885605096817 | {"E": 3} |
| part_same_1@2.0 | {"context association": 3} | 3/3 (1) | 0.3329186886548996 | {"E": 3} |
| part_same_1@3.0 | {"context association": 3} | 3/3 (1) | 0.3329186886548996 | {"E": 3} |
| part_same_1@4.0 | {"context association": 3} | 3/3 (1) | 0.3329186886548996 | {"E": 3} |
| part_same_2@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.00460655358619988 | {} |
| part_same_2@0.5 | {"context association": 1, "composition association": 2} | 1/3 (0.333333) | 0.02878849022090435 | {"D": 3} |
| part_same_2@1.0 | {"context association": 3} | 3/3 (1) | 0.35333065688610077 | {"D": 3} |
| part_same_2@2.0 | {"context association": 3} | 3/3 (1) | 0.47770389914512634 | {"D": 3} |
| part_same_2@3.0 | {"context association": 3} | 3/3 (1) | 0.47770388424396515 | {"D": 3} |
| part_same_2@4.0 | {"context association": 3} | 3/3 (1) | 0.47770389914512634 | {"D": 3} |
| part_same_3@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.019135822542011738 | {} |
| part_same_3@0.5 | {"composition association": 3} | 0/3 (0) | 0.04270814545452595 | {"R": 3} |
| part_same_3@1.0 | {"context association": 3} | 3/3 (1) | 0.23810721933841705 | {"R": 3} |
| part_same_3@2.0 | {"context association": 3} | 3/3 (1) | 0.36848604679107666 | {"R": 3} |
| part_same_3@3.0 | {"context association": 3} | 3/3 (1) | 0.36848604679107666 | {"R": 3} |
| part_same_3@4.0 | {"context association": 3} | 3/3 (1) | 0.36848604679107666 | {"R": 3} |
| part_same_4@0.0 | {"composition association": 1, "no association found by this procedure": 2} | 0/3 (0) | 0.007755723781883717 | {"H": 1} |
| part_same_4@0.5 | {"composition association": 2, "context association": 1} | 2/3 (0.666667) | 0.04552401416003704 | {"L": 3} |
| part_same_4@1.0 | {"context association": 3} | 3/3 (1) | 0.3066170960664749 | {"L": 3} |
| part_same_4@2.0 | {"context association": 3} | 3/3 (1) | 0.3066170960664749 | {"L": 3} |
| part_same_4@3.0 | {"context association": 3} | 3/3 (1) | 0.3066170960664749 | {"L": 3} |
| part_same_4@4.0 | {"context association": 3} | 3/3 (1) | 0.3066170960664749 | {"L": 3} |
| part_same_5@0.0 | {"no association found by this procedure": 2, "composition association": 1} | 0/3 (0) | 0.007350030122324824 | {"W": 1} |
| part_same_5@0.5 | {"context association": 2, "composition association": 1} | 3/3 (1) | 0.07248079776763916 | {"A": 3} |
| part_same_5@1.0 | {"context association": 3} | 3/3 (1) | 0.2120792344212532 | {"A": 3} |
| part_same_5@2.0 | {"context association": 3} | 3/3 (1) | 0.30696794390678406 | {"A": 3} |
| part_same_5@3.0 | {"context association": 3} | 3/3 (1) | 0.30696795880794525 | {"A": 3} |
| part_same_5@4.0 | {"context association": 3} | 3/3 (1) | 0.30696794390678406 | {"A": 3} |
| part_same_6@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.011297497898340225 | {"P": 1} |
| part_same_6@0.5 | {"composition association": 3} | 0/3 (0) | 0.01687772385776043 | {"Q": 3} |
| part_same_6@1.0 | {"context association": 3} | 3/3 (1) | 0.32063500583171844 | {"Q": 3} |
| part_same_6@2.0 | {"context association": 3} | 3/3 (1) | 0.35205410420894623 | {"Q": 3} |
| part_same_6@3.0 | {"context association": 3} | 3/3 (1) | 0.35205410420894623 | {"Q": 3} |
| part_same_6@4.0 | {"context association": 3} | 3/3 (1) | 0.35205410420894623 | {"Q": 3} |
| part_same_7@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.013886423781514168 | {} |
| part_same_7@0.5 | {"composition association": 3} | 1/3 (0.333333) | 0.01699833944439888 | {"S": 3} |
| part_same_7@1.0 | {"context association": 3} | 3/3 (1) | 0.3692713677883148 | {"S": 3} |
| part_same_7@2.0 | {"context association": 3} | 3/3 (1) | 0.497067853808403 | {"S": 3} |
| part_same_7@3.0 | {"context association": 3} | 3/3 (1) | 0.497067853808403 | {"S": 3} |
| part_same_7@4.0 | {"context association": 3} | 3/3 (1) | 0.497067853808403 | {"S": 3} |
| part_same_8@0.0 | {"no association found by this procedure": 2, "composition association": 1} | 0/3 (0) | 0.006091090152040124 | {"C": 1} |
| part_same_8@0.5 | {"composition association": 3} | 0/3 (0) | 0.02246236987411976 | {"T": 3} |
| part_same_8@1.0 | {"context association": 3} | 3/3 (1) | 0.3033718913793564 | {"T": 3} |
| part_same_8@2.0 | {"context association": 3} | 3/3 (1) | 0.4871852695941925 | {"T": 3} |
| part_same_8@3.0 | {"context association": 3} | 3/3 (1) | 0.4871852695941925 | {"T": 3} |
| part_same_8@4.0 | {"context association": 3} | 3/3 (1) | 0.4871852844953537 | {"T": 3} |
| part_same_9@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.02965017408132553 | {} |
| part_same_9@0.5 | {"composition association": 2, "no association found by this procedure": 1} | 0/3 (0) | 0.016070696525275707 | {"N": 3} |
| part_same_9@1.0 | {"context association": 3} | 3/3 (1) | 0.33912813663482666 | {"N": 3} |
| part_same_9@2.0 | {"context association": 3} | 3/3 (1) | 0.5058882385492325 | {"N": 3} |
| part_same_9@3.0 | {"context association": 3} | 3/3 (1) | 0.5058882385492325 | {"N": 3} |
| part_same_9@4.0 | {"context association": 3} | 3/3 (1) | 0.5058882385492325 | {"N": 3} |
| trp@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.0059544292744249105 | {"R": 1} |
| trp@0.5 | {"composition association": 3} | 0/3 (0) | 0.02085801213979721 | {"W": 3} |
| trp@1.0 | {"composition association": 3} | 0/3 (0) | 0.013635165989398956 | {"W": 3} |
| trp@2.0 | {"composition association": 3} | 0/3 (0) | 0.007197849452495575 | {"W": 3} |
| trp@3.0 | {"composition association": 3} | 0/3 (0) | 0.011202685534954071 | {"W": 3} |
| trp@4.0 | {"composition association": 3} | 0/3 (0) | 0.024296926334500313 | {"W": 3} |
| two_job@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.009068265557289124 | {} |
| two_job@0.5 | {"structural association": 3} | 3/3 (1) | 0.16162650287151337 | {} |
| two_job@1.0 | {"structural association": 3} | 3/3 (1) | 0.2512325644493103 | {} |
| two_job@2.0 | {"structural association": 3} | 3/3 (1) | 0.2360028624534607 | {"K": 1, "Y": 1} |
| two_job@3.0 | {"structural association": 3} | 3/3 (1) | 0.24781210720539093 | {"Y": 1} |
| two_job@4.0 | {"structural association": 3} | 3/3 (1) | 0.25861481577157974 | {} |

### sae:0.001

| Gate | Pass/total |
| --- | --- |
| buried | 12/12 |
| helix | 12/12 |
| hydro_win | 12/12 |
| lys_helix | 12/12 |
| narrow | 8/12 |
| noise + every alpha = 0 | 540/542 |
| ordered | 12/12 |
| trp | 12/12 |
| two_job | 12/12 |

Noise null: 3/518; rate 0.005791505791505791; 95% Clopper-Pearson [0.001195943889512063, 0.016831245598461497].

| Type@alpha | Calls | Structural k/n (share) | Median primary JSD | Residue modes |
| --- | --- | --- | --- | --- |
| buried@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.008296870160847902 | {} |
| buried@0.5 | {"structural association": 3} | 3/3 (1) | 0.2993399500846863 | {} |
| buried@1.0 | {"structural association": 3} | 3/3 (1) | 0.3543471544981003 | {} |
| buried@2.0 | {"structural association": 3} | 3/3 (1) | 0.3737981766462326 | {"I": 1} |
| buried@3.0 | {"structural association": 3} | 3/3 (1) | 0.3706211745738983 | {} |
| buried@4.0 | {"structural association": 3} | 3/3 (1) | 0.37910929322242737 | {} |
| helix@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.008748199790716171 | {} |
| helix@0.5 | {"structural association": 3} | 3/3 (1) | 0.2034878060221672 | {} |
| helix@1.0 | {"structural association": 3} | 3/3 (1) | 0.32258395105600357 | {} |
| helix@2.0 | {"structural association": 3} | 3/3 (1) | 0.387160986661911 | {} |
| helix@3.0 | {"structural association": 3} | 3/3 (1) | 0.39117099344730377 | {} |
| helix@4.0 | {"structural association": 3} | 3/3 (1) | 0.38399483263492584 | {} |
| hydro_win@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.004553650971502066 | {} |
| hydro_win@0.5 | {"no association found by this procedure": 3} | 0/3 (0) | 0.02244629245251417 | {"I": 2} |
| hydro_win@1.0 | {"context association": 2, "structural association": 1} | 3/3 (1) | 0.09869938343763351 | {"V": 2, "L": 2, "I": 1} |
| hydro_win@2.0 | {"context association": 3} | 3/3 (1) | 0.17408699542284012 | {"I": 3, "L": 3, "V": 2, "F": 1} |
| hydro_win@3.0 | {"context association": 3} | 3/3 (1) | 0.17299535125494003 | {"I": 3, "L": 2, "V": 3} |
| hydro_win@4.0 | {"context association": 3} | 3/3 (1) | 0.24444573372602463 | {"I": 3, "L": 3, "V": 3} |
| lys_helix@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.007483131717890501 | {"L": 1} |
| lys_helix@0.5 | {"composition association": 2, "no association found by this procedure": 1} | 0/3 (0) | 0.014395940117537975 | {"K": 3} |
| lys_helix@1.0 | {"context association": 3} | 3/3 (1) | 0.11597701534628868 | {"K": 3} |
| lys_helix@2.0 | {"context association": 3} | 3/3 (1) | 0.38526657223701477 | {"K": 3} |
| lys_helix@3.0 | {"context association": 3} | 3/3 (1) | 0.38526657223701477 | {"K": 3} |
| lys_helix@4.0 | {"context association": 3} | 3/3 (1) | 0.38526657223701477 | {"K": 3} |
| mix_diff_20@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.004976521013304591 | {"S": 1} |
| mix_diff_20@0.5 | {"no association found by this procedure": 3} | 0/3 (0) | 0.03953410126268864 | {"Q": 1, "R": 1} |
| mix_diff_20@1.0 | {"structural association": 2, "no association found by this procedure": 1} | 3/3 (1) | 0.07537580281496048 | {} |
| mix_diff_20@2.0 | {"structural association": 3} | 3/3 (1) | 0.09474439732730389 | {"N": 1, "K": 1, "Q": 1} |
| mix_diff_20@3.0 | {"context association": 1, "structural association": 2} | 3/3 (1) | 0.09290515258908272 | {"D": 1, "K": 1, "R": 1} |
| mix_diff_20@4.0 | {"structural association": 3} | 3/3 (1) | 0.0805644728243351 | {} |
| mix_diff_2@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.019224680960178375 | {} |
| mix_diff_2@0.5 | {"no association found by this procedure": 1, "composition association": 2} | 0/3 (0) | 0.024875488132238388 | {"K": 2} |
| mix_diff_2@1.0 | {"context association": 3} | 3/3 (1) | 0.09322240203619003 | {"E": 2, "K": 3} |
| mix_diff_2@2.0 | {"context association": 3} | 3/3 (1) | 0.27136803418397903 | {"E": 3, "K": 3} |
| mix_diff_2@3.0 | {"context association": 3} | 3/3 (1) | 0.27136801928281784 | {"E": 3, "K": 3} |
| mix_diff_2@4.0 | {"context association": 3} | 3/3 (1) | 0.27136801928281784 | {"E": 3, "K": 3} |
| mix_diff_8@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.011675784830003977 | {"M": 1} |
| mix_diff_8@0.5 | {"composition association": 1, "no association found by this procedure": 2} | 1/3 (0.333333) | 0.040124788880348206 | {"D": 3, "Q": 1, "R": 1} |
| mix_diff_8@1.0 | {"context association": 3} | 3/3 (1) | 0.0938503947108984 | {"K": 1, "Q": 3, "R": 2, "A": 1, "D": 1} |
| mix_diff_8@2.0 | {"context association": 3} | 3/3 (1) | 0.10320716351270676 | {"D": 3, "K": 2, "Q": 3, "R": 2} |
| mix_diff_8@3.0 | {"context association": 3} | 3/3 (1) | 0.08919289335608482 | {"A": 1, "D": 3, "K": 3, "R": 2, "Q": 2} |
| mix_diff_8@4.0 | {"context association": 3} | 3/3 (1) | 0.08991399966180325 | {"D": 3, "Q": 2, "R": 1, "K": 1} |
| mix_same_20@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.00823838310316205 | {} |
| mix_same_20@0.5 | {"structural association": 3} | 3/3 (1) | 0.226395845413208 | {"W": 1} |
| mix_same_20@1.0 | {"structural association": 3} | 3/3 (1) | 0.32353703677654266 | {} |
| mix_same_20@2.0 | {"structural association": 3} | 3/3 (1) | 0.389296293258667 | {} |
| mix_same_20@3.0 | {"structural association": 3} | 3/3 (1) | 0.3822653591632843 | {} |
| mix_same_20@4.0 | {"structural association": 3} | 3/3 (1) | 0.38779768347740173 | {"E": 1} |
| mix_same_2@0.0 | {"no association found by this procedure": 3} | 1/3 (0.333333) | 0.006883922265842557 | {} |
| mix_same_2@0.5 | {"no association found by this procedure": 1, "composition association": 2} | 1/3 (0.333333) | 0.035290902480483055 | {"E": 2, "K": 3, "P": 1} |
| mix_same_2@1.0 | {"context association": 3} | 3/3 (1) | 0.15516679733991623 | {"E": 3, "K": 3} |
| mix_same_2@2.0 | {"context association": 3} | 3/3 (1) | 0.3544857054948807 | {"E": 3, "K": 3} |
| mix_same_2@3.0 | {"context association": 3} | 3/3 (1) | 0.35286320000886917 | {"E": 3, "K": 3} |
| mix_same_2@4.0 | {"context association": 3} | 3/3 (1) | 0.3552985042333603 | {"E": 3, "K": 3} |
| mix_same_8@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.01656216010451317 | {"G": 1} |
| mix_same_8@0.5 | {"structural association": 3} | 3/3 (1) | 0.11358334124088287 | {"A": 1} |
| mix_same_8@1.0 | {"structural association": 2, "context association": 1} | 3/3 (1) | 0.28477753698825836 | {"R": 2, "Q": 1, "E": 1} |
| mix_same_8@2.0 | {"structural association": 3} | 3/3 (1) | 0.35176242887973785 | {"E": 2, "S": 1} |
| mix_same_8@3.0 | {"structural association": 3} | 3/3 (1) | 0.3618476986885071 | {"A": 1, "D": 2, "L": 1, "Q": 1} |
| mix_same_8@4.0 | {"structural association": 2, "context association": 1} | 3/3 (1) | 0.3536161333322525 | {"E": 1, "L": 1, "A": 1} |
| narrow@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.009567268192768097 | {"C": 1} |
| narrow@0.5 | {"no association found by this procedure": 3} | 0/3 (0) | 0.031947758980095387 | {"T": 3, "S": 2} |
| narrow@1.0 | {"context association": 3} | 3/3 (1) | 0.07796252146363258 | {"S": 3, "T": 3} |
| narrow@2.0 | {"context association": 3} | 3/3 (1) | 0.46556901931762695 | {"S": 3, "T": 3} |
| narrow@3.0 | {"context association": 3} | 3/3 (1) | 0.49249857664108276 | {"S": 3, "T": 3} |
| narrow@4.0 | {"context association": 3} | 3/3 (1) | 0.49226078391075134 | {"S": 3, "T": 3} |
| noise@0.0 | {"no association found by this procedure": 501, "structural association": 1, "composition association": 1} | 3/503 (0.00596421) | 0.011400014162063599 | {"A": 3, "D": 7, "Q": 5, "C": 2, "H": 3, "Y": 2, "S": 4, "R": 3, "N": 5, "T": 2, "G": 2, "L": 2, "W": 2, "V": 3, "E": 2, "K": 1, "P": 1, "I": 2, "M": 1} |
| noise@0.5 | {"no association found by this procedure": 3} | 0/3 (0) | 0.011916222050786018 | {} |
| noise@1.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.02299284003674984 | {"G": 1} |
| noise@2.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.018197812139987946 | {} |
| noise@3.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.006964396685361862 | {} |
| noise@4.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.0036002397537231445 | {} |
| ordered@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.005336402915418148 | {} |
| ordered@0.5 | {"structural association": 3} | 3/3 (1) | 0.1420007236301899 | {} |
| ordered@1.0 | {"structural association": 3} | 3/3 (1) | 0.2153334990143776 | {"A": 1} |
| ordered@2.0 | {"structural association": 3} | 3/3 (1) | 0.25158096849918365 | {} |
| ordered@3.0 | {"structural association": 3} | 3/3 (1) | 0.26180223375558853 | {} |
| ordered@4.0 | {"structural association": 3} | 3/3 (1) | 0.25393153727054596 | {"D": 1} |
| part_diff_0@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.006365666165947914 | {"S": 1} |
| part_diff_0@0.5 | {"composition association": 3} | 1/3 (0.333333) | 0.031458109617233276 | {"K": 3} |
| part_diff_0@1.0 | {"context association": 3} | 3/3 (1) | 0.12964210659265518 | {"K": 3} |
| part_diff_0@2.0 | {"context association": 3} | 3/3 (1) | 0.38513487577438354 | {"K": 3} |
| part_diff_0@3.0 | {"context association": 3} | 3/3 (1) | 0.38526657223701477 | {"K": 3} |
| part_diff_0@4.0 | {"context association": 3} | 3/3 (1) | 0.38526657223701477 | {"K": 3} |
| part_diff_10@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.020702771842479706 | {"N": 1} |
| part_diff_10@0.5 | {"composition association": 3} | 0/3 (0) | 0.012641368433833122 | {"R": 1, "V": 3} |
| part_diff_10@1.0 | {"context association": 3} | 3/3 (1) | 0.14238285273313522 | {"V": 3} |
| part_diff_10@2.0 | {"context association": 3} | 3/3 (1) | 0.4628424197435379 | {"V": 3} |
| part_diff_10@3.0 | {"context association": 3} | 3/3 (1) | 0.4628424197435379 | {"V": 3} |
| part_diff_10@4.0 | {"context association": 3} | 3/3 (1) | 0.4628424197435379 | {"V": 3} |
| part_diff_11@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.00851874565705657 | {} |
| part_diff_11@0.5 | {"no association found by this procedure": 3} | 0/3 (0) | 0.008793617133051157 | {"I": 1, "G": 1} |
| part_diff_11@1.0 | {"context association": 3} | 3/3 (1) | 0.11534936726093292 | {"I": 3} |
| part_diff_11@2.0 | {"context association": 3} | 3/3 (1) | 0.6163517087697983 | {"I": 3} |
| part_diff_11@3.0 | {"context association": 3} | 3/3 (1) | 0.6225050836801529 | {"I": 3} |
| part_diff_11@4.0 | {"context association": 3} | 3/3 (1) | 0.6225050836801529 | {"I": 3} |
| part_diff_12@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.013218275271356106 | {} |
| part_diff_12@0.5 | {"no association found by this procedure": 3} | 0/3 (0) | 0.018385962583124638 | {"G": 3} |
| part_diff_12@1.0 | {"context association": 3} | 3/3 (1) | 0.13921713083982468 | {"G": 3} |
| part_diff_12@2.0 | {"context association": 3} | 3/3 (1) | 0.5942998230457306 | {"G": 3} |
| part_diff_12@3.0 | {"context association": 3} | 3/3 (1) | 0.6001782417297363 | {"G": 3} |
| part_diff_12@4.0 | {"context association": 3} | 3/3 (1) | 0.6001782417297363 | {"G": 3} |
| part_diff_13@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.005927003920078278 | {} |
| part_diff_13@0.5 | {"no association found by this procedure": 3} | 0/3 (0) | 0.009839633014053106 | {"F": 1, "K": 1} |
| part_diff_13@1.0 | {"composition association": 1, "context association": 2} | 3/3 (1) | 0.0868336446583271 | {"F": 3} |
| part_diff_13@2.0 | {"context association": 3} | 3/3 (1) | 0.5717944651842117 | {"F": 3} |
| part_diff_13@3.0 | {"context association": 3} | 3/3 (1) | 0.5717944651842117 | {"F": 3} |
| part_diff_13@4.0 | {"context association": 3} | 3/3 (1) | 0.5717944651842117 | {"F": 3} |
| part_diff_14@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.008658186998218298 | {"D": 1} |
| part_diff_14@0.5 | {"no association found by this procedure": 2, "composition association": 1} | 0/3 (0) | 0.010429229121655226 | {"Y": 2} |
| part_diff_14@1.0 | {"composition association": 3} | 1/3 (0.333333) | 0.0381165724247694 | {"Y": 3} |
| part_diff_14@2.0 | {"context association": 3} | 3/3 (1) | 0.5403773188591003 | {"Y": 3} |
| part_diff_14@3.0 | {"context association": 3} | 3/3 (1) | 0.5446189641952515 | {"Y": 3} |
| part_diff_14@4.0 | {"context association": 3} | 3/3 (1) | 0.5446189641952515 | {"Y": 3} |
| part_diff_15@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.01081624859943986 | {} |
| part_diff_15@0.5 | {"no association found by this procedure": 2, "composition association": 1} | 0/3 (0) | 0.024507855996489525 | {"H": 1} |
| part_diff_15@1.0 | {"composition association": 1, "context association": 2} | 2/3 (0.666667) | 0.05821920931339264 | {"H": 3} |
| part_diff_15@2.0 | {"context association": 3} | 3/3 (1) | 0.41038843989372253 | {"H": 3} |
| part_diff_15@3.0 | {"context association": 3} | 3/3 (1) | 0.4525968134403229 | {"H": 3} |
| part_diff_15@4.0 | {"context association": 3} | 3/3 (1) | 0.4525968134403229 | {"H": 3} |
| part_diff_16@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.023377791047096252 | {"F": 1} |
| part_diff_16@0.5 | {"no association found by this procedure": 3} | 0/3 (0) | 0.00814689276739955 | {} |
| part_diff_16@1.0 | {"composition association": 3} | 0/3 (0) | 0.014837131835520267 | {"M": 3} |
| part_diff_16@2.0 | {"context association": 3} | 3/3 (1) | 0.5011963099241257 | {"M": 3} |
| part_diff_16@3.0 | {"context association": 3} | 3/3 (1) | 0.6780321896076202 | {"M": 3} |
| part_diff_16@4.0 | {"context association": 3} | 3/3 (1) | 0.6780321896076202 | {"M": 3} |
| part_diff_17@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.00610646978020668 | {} |
| part_diff_17@0.5 | {"no association found by this procedure": 3} | 0/3 (0) | 0.0038901595398783684 | {"K": 1, "W": 1} |
| part_diff_17@1.0 | {"composition association": 3} | 0/3 (0) | 0.0233630845323205 | {"W": 3} |
| part_diff_17@2.0 | {"context association": 3} | 3/3 (1) | 0.44683730602264404 | {"W": 3} |
| part_diff_17@3.0 | {"context association": 3} | 3/3 (1) | 0.5702565014362335 | {"W": 3} |
| part_diff_17@4.0 | {"context association": 3} | 3/3 (1) | 0.5702565014362335 | {"W": 3} |
| part_diff_18@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.010289493948221207 | {} |
| part_diff_18@0.5 | {"no association found by this procedure": 3} | 0/3 (0) | 0.014642532914876938 | {} |
| part_diff_18@1.0 | {"composition association": 3} | 0/3 (0) | 0.013980010524392128 | {"C": 3} |
| part_diff_18@2.0 | {"context association": 3} | 3/3 (1) | 0.4312226474285126 | {"C": 3} |
| part_diff_18@3.0 | {"context association": 3} | 3/3 (1) | 0.4757852256298065 | {"C": 3} |
| part_diff_18@4.0 | {"context association": 3} | 3/3 (1) | 0.4757852256298065 | {"C": 3} |
| part_diff_19@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.011696331202983856 | {} |
| part_diff_19@0.5 | {"no association found by this procedure": 3} | 0/3 (0) | 0.028556447476148605 | {"P": 1} |
| part_diff_19@1.0 | {"composition association": 2, "context association": 1} | 1/3 (0.333333) | 0.03179760091006756 | {"P": 3} |
| part_diff_19@2.0 | {"context association": 3} | 3/3 (1) | 0.6861454248428345 | {"P": 3} |
| part_diff_19@3.0 | {"context association": 3} | 3/3 (1) | 0.7920278012752533 | {"P": 3} |
| part_diff_19@4.0 | {"context association": 3} | 3/3 (1) | 0.7920278012752533 | {"P": 3} |
| part_diff_1@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.008448036387562752 | {} |
| part_diff_1@0.5 | {"no association found by this procedure": 3} | 0/3 (0) | 0.00809803418815136 | {} |
| part_diff_1@1.0 | {"context association": 3} | 3/3 (1) | 0.04823595844209194 | {"E": 3} |
| part_diff_1@2.0 | {"context association": 3} | 3/3 (1) | 0.602769136428833 | {"E": 3} |
| part_diff_1@3.0 | {"context association": 3} | 3/3 (1) | 0.7306002974510193 | {"E": 3} |
| part_diff_1@4.0 | {"context association": 3} | 3/3 (1) | 0.7306002974510193 | {"E": 3} |
| part_diff_2@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.009566405788064003 | {} |
| part_diff_2@0.5 | {"composition association": 2, "no association found by this procedure": 1} | 0/3 (0) | 0.020873794332146645 | {"D": 3, "G": 1} |
| part_diff_2@1.0 | {"composition association": 1, "context association": 2} | 2/3 (0.666667) | 0.10241330415010452 | {"D": 3} |
| part_diff_2@2.0 | {"context association": 3} | 3/3 (1) | 0.37018750607967377 | {"D": 3} |
| part_diff_2@3.0 | {"context association": 3} | 3/3 (1) | 0.37018750607967377 | {"D": 3} |
| part_diff_2@4.0 | {"context association": 3} | 3/3 (1) | 0.37018750607967377 | {"D": 3} |
| part_diff_3@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.006683044834062457 | {} |
| part_diff_3@0.5 | {"composition association": 3} | 1/3 (0.333333) | 0.00851841876283288 | {"R": 3} |
| part_diff_3@1.0 | {"context association": 3} | 3/3 (1) | 0.14965280145406723 | {"R": 3} |
| part_diff_3@2.0 | {"context association": 3} | 3/3 (1) | 0.36747829616069794 | {"R": 3} |
| part_diff_3@3.0 | {"context association": 3} | 3/3 (1) | 0.36848604679107666 | {"R": 3} |
| part_diff_3@4.0 | {"context association": 3} | 3/3 (1) | 0.36848604679107666 | {"R": 3} |
| part_diff_4@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.0164029560983181 | {} |
| part_diff_4@0.5 | {"no association found by this procedure": 3} | 0/3 (0) | 0.013631873298436403 | {} |
| part_diff_4@1.0 | {"context association": 3} | 3/3 (1) | 0.1623123660683632 | {"L": 3} |
| part_diff_4@2.0 | {"context association": 3} | 3/3 (1) | 0.577381044626236 | {"L": 3} |
| part_diff_4@3.0 | {"context association": 3} | 3/3 (1) | 0.640703946352005 | {"L": 3} |
| part_diff_4@4.0 | {"context association": 3} | 3/3 (1) | 0.640703946352005 | {"L": 3} |
| part_diff_5@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.015131203457713127 | {} |
| part_diff_5@0.5 | {"no association found by this procedure": 3} | 0/3 (0) | 0.016403728164732456 | {"A": 2} |
| part_diff_5@1.0 | {"context association": 3} | 3/3 (1) | 0.1376957818865776 | {"A": 3} |
| part_diff_5@2.0 | {"context association": 3} | 3/3 (1) | 0.5838223546743393 | {"A": 3} |
| part_diff_5@3.0 | {"context association": 3} | 3/3 (1) | 0.5838223546743393 | {"A": 3} |
| part_diff_5@4.0 | {"context association": 3} | 3/3 (1) | 0.5838223546743393 | {"A": 3} |
| part_diff_6@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.01102103665471077 | {"D": 1} |
| part_diff_6@0.5 | {"composition association": 3} | 0/3 (0) | 0.006539138033986092 | {"Q": 3} |
| part_diff_6@1.0 | {"context association": 2, "composition association": 1} | 2/3 (0.666667) | 0.06628570705652237 | {"Q": 3} |
| part_diff_6@2.0 | {"context association": 3} | 3/3 (1) | 0.31640857458114624 | {"Q": 3} |
| part_diff_6@3.0 | {"context association": 3} | 3/3 (1) | 0.35205410420894623 | {"Q": 3} |
| part_diff_6@4.0 | {"context association": 3} | 3/3 (1) | 0.35205410420894623 | {"Q": 3} |
| part_diff_7@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.007304178550839424 | {"V": 1} |
| part_diff_7@0.5 | {"no association found by this procedure": 3} | 0/3 (0) | 0.01936476305127144 | {"N": 1} |
| part_diff_7@1.0 | {"context association": 3} | 3/3 (1) | 0.10416379943490028 | {"S": 3} |
| part_diff_7@2.0 | {"context association": 3} | 3/3 (1) | 0.6331854164600372 | {"S": 3} |
| part_diff_7@3.0 | {"context association": 3} | 3/3 (1) | 0.7237382233142853 | {"S": 3} |
| part_diff_7@4.0 | {"context association": 3} | 3/3 (1) | 0.7237382233142853 | {"S": 3} |
| part_diff_8@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.010969500988721848 | {"G": 1} |
| part_diff_8@0.5 | {"no association found by this procedure": 3} | 0/3 (0) | 0.00841737911105156 | {"T": 3} |
| part_diff_8@1.0 | {"context association": 3} | 3/3 (1) | 0.12359160929918289 | {"T": 3} |
| part_diff_8@2.0 | {"context association": 3} | 3/3 (1) | 0.47146621346473694 | {"T": 3} |
| part_diff_8@3.0 | {"context association": 3} | 3/3 (1) | 0.4714662879705429 | {"T": 3} |
| part_diff_8@4.0 | {"context association": 3} | 3/3 (1) | 0.4714662879705429 | {"T": 3} |
| part_diff_9@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.01201760396361351 | {"Q": 1} |
| part_diff_9@0.5 | {"no association found by this procedure": 3} | 0/3 (0) | 0.02185926493257284 | {"N": 2} |
| part_diff_9@1.0 | {"composition association": 1, "context association": 2} | 2/3 (0.666667) | 0.11814187094569206 | {"N": 3} |
| part_diff_9@2.0 | {"context association": 3} | 3/3 (1) | 0.5034876018762589 | {"N": 3} |
| part_diff_9@3.0 | {"context association": 3} | 3/3 (1) | 0.5058882385492325 | {"N": 3} |
| part_diff_9@4.0 | {"context association": 3} | 3/3 (1) | 0.5058882385492325 | {"N": 3} |
| part_same_0@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.01760019361972809 | {} |
| part_same_0@0.5 | {"no association found by this procedure": 2, "composition association": 1} | 0/3 (0) | 0.02284758724272251 | {"K": 3} |
| part_same_0@1.0 | {"composition association": 1, "context association": 2} | 3/3 (1) | 0.07141196355223656 | {"K": 3} |
| part_same_0@2.0 | {"context association": 3} | 3/3 (1) | 0.3852665573358536 | {"K": 3} |
| part_same_0@3.0 | {"context association": 3} | 3/3 (1) | 0.38526657223701477 | {"K": 3} |
| part_same_0@4.0 | {"context association": 3} | 3/3 (1) | 0.38526657223701477 | {"K": 3} |
| part_same_10@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.007556013995781541 | {} |
| part_same_10@0.5 | {"no association found by this procedure": 2, "composition association": 1} | 0/3 (0) | 0.02080779615789652 | {"V": 2} |
| part_same_10@1.0 | {"context association": 3} | 3/3 (1) | 0.13608945906162262 | {"V": 3} |
| part_same_10@2.0 | {"context association": 3} | 3/3 (1) | 0.4317833483219147 | {"V": 3} |
| part_same_10@3.0 | {"context association": 3} | 3/3 (1) | 0.4317833483219147 | {"V": 3} |
| part_same_10@4.0 | {"context association": 3} | 3/3 (1) | 0.4317833483219147 | {"V": 3} |
| part_same_11@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.004447488114237785 | {} |
| part_same_11@0.5 | {"composition association": 2, "no association found by this procedure": 1} | 0/3 (0) | 0.01241509709507227 | {"I": 3} |
| part_same_11@1.0 | {"context association": 3} | 3/3 (1) | 0.13923075422644615 | {"I": 3} |
| part_same_11@2.0 | {"context association": 3} | 3/3 (1) | 0.3694564551115036 | {"I": 3} |
| part_same_11@3.0 | {"context association": 3} | 3/3 (1) | 0.3694564551115036 | {"I": 3} |
| part_same_11@4.0 | {"context association": 3} | 3/3 (1) | 0.3694564551115036 | {"I": 3} |
| part_same_12@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.014452284667640924 | {} |
| part_same_12@0.5 | {"composition association": 1, "no association found by this procedure": 2} | 0/3 (0) | 0.013252112548798323 | {"G": 2} |
| part_same_12@1.0 | {"context association": 3} | 3/3 (1) | 0.16877296566963196 | {"G": 3} |
| part_same_12@2.0 | {"context association": 3} | 3/3 (1) | 0.5688328146934509 | {"G": 3} |
| part_same_12@3.0 | {"context association": 3} | 3/3 (1) | 0.6001782417297363 | {"G": 3} |
| part_same_12@4.0 | {"context association": 3} | 3/3 (1) | 0.6001782417297363 | {"G": 3} |
| part_same_13@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.01816152222454548 | {} |
| part_same_13@0.5 | {"no association found by this procedure": 3} | 0/3 (0) | 0.013742534909397364 | {"F": 2} |
| part_same_13@1.0 | {"composition association": 1, "context association": 2} | 3/3 (1) | 0.06930900923907757 | {"F": 3} |
| part_same_13@2.0 | {"context association": 3} | 3/3 (1) | 0.3401835560798645 | {"F": 3} |
| part_same_13@3.0 | {"context association": 3} | 3/3 (1) | 0.3770249933004379 | {"F": 3} |
| part_same_13@4.0 | {"context association": 3} | 3/3 (1) | 0.37702496349811554 | {"F": 3} |
| part_same_14@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.017748580314218998 | {} |
| part_same_14@0.5 | {"no association found by this procedure": 2, "composition association": 1} | 0/3 (0) | 0.010938577353954315 | {"Y": 3} |
| part_same_14@1.0 | {"composition association": 2, "context association": 1} | 2/3 (0.666667) | 0.043143343180418015 | {"Y": 3} |
| part_same_14@2.0 | {"context association": 3} | 3/3 (1) | 0.39137980341911316 | {"Y": 3} |
| part_same_14@3.0 | {"context association": 3} | 3/3 (1) | 0.39137980341911316 | {"Y": 3} |
| part_same_14@4.0 | {"context association": 3} | 3/3 (1) | 0.39137980341911316 | {"Y": 3} |
| part_same_15@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.012078006751835346 | {} |
| part_same_15@0.5 | {"no association found by this procedure": 3} | 0/3 (0) | 0.007519897073507309 | {"D": 1, "H": 1, "S": 1} |
| part_same_15@1.0 | {"composition association": 1, "context association": 2} | 2/3 (0.666667) | 0.05733689293265343 | {"H": 3} |
| part_same_15@2.0 | {"context association": 3} | 3/3 (1) | 0.45157676935195923 | {"H": 3} |
| part_same_15@3.0 | {"context association": 3} | 3/3 (1) | 0.45259687304496765 | {"H": 3} |
| part_same_15@4.0 | {"context association": 3} | 3/3 (1) | 0.4525968134403229 | {"H": 3} |
| part_same_16@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.005089645506814122 | {"W": 1} |
| part_same_16@0.5 | {"no association found by this procedure": 2, "composition association": 1} | 0/3 (0) | 0.015595378819853067 | {"M": 2} |
| part_same_16@1.0 | {"context association": 1, "composition association": 2} | 3/3 (1) | 0.054969895631074905 | {"M": 3} |
| part_same_16@2.0 | {"context association": 3} | 3/3 (1) | 0.33179551362991333 | {"M": 3} |
| part_same_16@3.0 | {"context association": 3} | 3/3 (1) | 0.3695078194141388 | {"M": 3} |
| part_same_16@4.0 | {"context association": 3} | 3/3 (1) | 0.3695078194141388 | {"M": 3} |
| part_same_17@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.0020545031875371933 | {} |
| part_same_17@0.5 | {"no association found by this procedure": 3} | 0/3 (0) | 0.005786853609606624 | {"W": 1} |
| part_same_17@1.0 | {"context association": 1, "composition association": 2} | 1/3 (0.333333) | 0.04384882561862469 | {"W": 3} |
| part_same_17@2.0 | {"context association": 3} | 3/3 (1) | 0.26885080337524414 | {"W": 3} |
| part_same_17@3.0 | {"context association": 3} | 3/3 (1) | 0.33689049631357193 | {"W": 3} |
| part_same_17@4.0 | {"context association": 3} | 3/3 (1) | 0.33689049631357193 | {"W": 3} |
| part_same_18@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.016077551990747452 | {} |
| part_same_18@0.5 | {"no association found by this procedure": 3} | 0/3 (0) | 0.009316746611148119 | {"C": 2} |
| part_same_18@1.0 | {"composition association": 3} | 0/3 (0) | 0.018149160780012608 | {"C": 3} |
| part_same_18@2.0 | {"context association": 3} | 3/3 (1) | 0.374982625246048 | {"C": 3} |
| part_same_18@3.0 | {"context association": 3} | 3/3 (1) | 0.4757852256298065 | {"C": 3} |
| part_same_18@4.0 | {"context association": 3} | 3/3 (1) | 0.4757852256298065 | {"C": 3} |
| part_same_19@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.009006621781736612 | {} |
| part_same_19@0.5 | {"composition association": 1, "no association found by this procedure": 2} | 0/3 (0) | 0.01915847882628441 | {"P": 1} |
| part_same_19@1.0 | {"context association": 3} | 3/3 (1) | 0.04509328678250313 | {"P": 3} |
| part_same_19@2.0 | {"context association": 3} | 3/3 (1) | 0.6315754950046539 | {"P": 3} |
| part_same_19@3.0 | {"context association": 3} | 3/3 (1) | 0.6315754950046539 | {"P": 3} |
| part_same_19@4.0 | {"context association": 3} | 3/3 (1) | 0.6315754950046539 | {"P": 3} |
| part_same_1@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.008413410745561123 | {} |
| part_same_1@0.5 | {"composition association": 3} | 0/3 (0) | 0.04064268618822098 | {"E": 3} |
| part_same_1@1.0 | {"context association": 3} | 3/3 (1) | 0.1467696875333786 | {"E": 3} |
| part_same_1@2.0 | {"context association": 3} | 3/3 (1) | 0.3329186886548996 | {"E": 3} |
| part_same_1@3.0 | {"context association": 3} | 3/3 (1) | 0.3329186886548996 | {"E": 3} |
| part_same_1@4.0 | {"context association": 3} | 3/3 (1) | 0.3329186886548996 | {"E": 3} |
| part_same_2@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.01136280968785286 | {"A": 1} |
| part_same_2@0.5 | {"no association found by this procedure": 3} | 0/3 (0) | 0.03078772872686386 | {"D": 3} |
| part_same_2@1.0 | {"context association": 3} | 3/3 (1) | 0.1431903913617134 | {"D": 3} |
| part_same_2@2.0 | {"context association": 3} | 3/3 (1) | 0.47770389914512634 | {"D": 3} |
| part_same_2@3.0 | {"context association": 3} | 3/3 (1) | 0.47770389914512634 | {"D": 3} |
| part_same_2@4.0 | {"context association": 3} | 3/3 (1) | 0.47770389914512634 | {"D": 3} |
| part_same_3@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.010801190976053476 | {"T": 1} |
| part_same_3@0.5 | {"no association found by this procedure": 3} | 0/3 (0) | 0.023590413853526115 | {"F": 1, "R": 3} |
| part_same_3@1.0 | {"context association": 2, "composition association": 1} | 2/3 (0.666667) | 0.0930028110742569 | {"R": 3} |
| part_same_3@2.0 | {"context association": 3} | 3/3 (1) | 0.36848604679107666 | {"R": 3} |
| part_same_3@3.0 | {"context association": 3} | 3/3 (1) | 0.36848604679107666 | {"R": 3} |
| part_same_3@4.0 | {"context association": 3} | 3/3 (1) | 0.36848604679107666 | {"R": 3} |
| part_same_4@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.00808686576783657 | {} |
| part_same_4@0.5 | {"no association found by this procedure": 1, "composition association": 2} | 0/3 (0) | 0.02986256405711174 | {"L": 3} |
| part_same_4@1.0 | {"context association": 3} | 3/3 (1) | 0.12008601427078247 | {"L": 3} |
| part_same_4@2.0 | {"context association": 3} | 3/3 (1) | 0.3066170960664749 | {"L": 3} |
| part_same_4@3.0 | {"context association": 3} | 3/3 (1) | 0.3066170960664749 | {"L": 3} |
| part_same_4@4.0 | {"context association": 3} | 3/3 (1) | 0.3066170960664749 | {"L": 3} |
| part_same_5@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.017168257385492325 | {} |
| part_same_5@0.5 | {"composition association": 3} | 0/3 (0) | 0.02911958284676075 | {"A": 3} |
| part_same_5@1.0 | {"context association": 3} | 3/3 (1) | 0.16165180131793022 | {"A": 3} |
| part_same_5@2.0 | {"context association": 3} | 3/3 (1) | 0.30696794390678406 | {"A": 3} |
| part_same_5@3.0 | {"context association": 3} | 3/3 (1) | 0.30696794390678406 | {"A": 3} |
| part_same_5@4.0 | {"context association": 3} | 3/3 (1) | 0.30696794390678406 | {"A": 3} |
| part_same_6@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.011246388778090477 | {} |
| part_same_6@0.5 | {"composition association": 2, "no association found by this procedure": 1} | 0/3 (0) | 0.007095332955941558 | {"Q": 3, "E": 1, "N": 1} |
| part_same_6@1.0 | {"context association": 2, "composition association": 1} | 3/3 (1) | 0.06355850212275982 | {"Q": 3} |
| part_same_6@2.0 | {"context association": 3} | 3/3 (1) | 0.35205410420894623 | {"Q": 3} |
| part_same_6@3.0 | {"context association": 3} | 3/3 (1) | 0.35205410420894623 | {"Q": 3} |
| part_same_6@4.0 | {"context association": 3} | 3/3 (1) | 0.35205410420894623 | {"Q": 3} |
| part_same_7@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.009409415069967508 | {} |
| part_same_7@0.5 | {"no association found by this procedure": 1, "composition association": 2} | 0/3 (0) | 0.03416907973587513 | {"S": 3} |
| part_same_7@1.0 | {"context association": 3} | 3/3 (1) | 0.12312699109315872 | {"S": 3} |
| part_same_7@2.0 | {"context association": 3} | 3/3 (1) | 0.497067853808403 | {"S": 3} |
| part_same_7@3.0 | {"context association": 3} | 3/3 (1) | 0.497067853808403 | {"S": 3} |
| part_same_7@4.0 | {"context association": 3} | 3/3 (1) | 0.497067853808403 | {"S": 3} |
| part_same_8@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.004025638103485107 | {} |
| part_same_8@0.5 | {"no association found by this procedure": 2, "composition association": 1} | 0/3 (0) | 0.028964277356863022 | {"T": 3} |
| part_same_8@1.0 | {"context association": 3} | 3/3 (1) | 0.09724307060241699 | {"T": 3} |
| part_same_8@2.0 | {"context association": 3} | 3/3 (1) | 0.4871852695941925 | {"T": 3} |
| part_same_8@3.0 | {"context association": 3} | 3/3 (1) | 0.4871852844953537 | {"T": 3} |
| part_same_8@4.0 | {"context association": 3} | 3/3 (1) | 0.4871852844953537 | {"T": 3} |
| part_same_9@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.00753465062007308 | {} |
| part_same_9@0.5 | {"no association found by this procedure": 3} | 0/3 (0) | 0.0199650377035141 | {"N": 2} |
| part_same_9@1.0 | {"context association": 3} | 3/3 (1) | 0.07343733869493008 | {"N": 3} |
| part_same_9@2.0 | {"context association": 3} | 3/3 (1) | 0.4735344350337982 | {"N": 3} |
| part_same_9@3.0 | {"context association": 3} | 3/3 (1) | 0.5058882385492325 | {"N": 3} |
| part_same_9@4.0 | {"context association": 3} | 3/3 (1) | 0.5058882385492325 | {"N": 3} |
| trp@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.013307890854775906 | {} |
| trp@0.5 | {"composition association": 2, "no association found by this procedure": 1} | 0/3 (0) | 0.01200861344113946 | {"W": 3} |
| trp@1.0 | {"composition association": 3} | 0/3 (0) | 0.008857954759150743 | {"W": 3} |
| trp@2.0 | {"composition association": 3} | 0/3 (0) | 0.009327371139079332 | {"W": 3} |
| trp@3.0 | {"composition association": 3} | 0/3 (0) | 0.011955339461565018 | {"W": 3} |
| trp@4.0 | {"composition association": 3} | 0/3 (0) | 0.017465275712311268 | {"W": 3} |
| two_job@0.0 | {"no association found by this procedure": 3} | 0/3 (0) | 0.014502506237477064 | {"P": 1} |
| two_job@0.5 | {"structural association": 3} | 3/3 (1) | 0.11868150532245636 | {} |
| two_job@1.0 | {"structural association": 3} | 3/3 (1) | 0.23900900781154633 | {"Y": 1} |
| two_job@2.0 | {"structural association": 3} | 3/3 (1) | 0.2513410896062851 | {"T": 1, "W": 1} |
| two_job@3.0 | {"structural association": 3} | 3/3 (1) | 0.2525147348642349 | {"K": 1} |
| two_job@4.0 | {"structural association": 3} | 3/3 (1) | 0.24926063418388367 | {"K": 1, "M": 1} |


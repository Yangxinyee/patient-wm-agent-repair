Forward pass 23 covered the last 25 non-test stays plus the 75 test stays, and cost only +0.0002.

| Stay set | Readings | Cost of mk2(2.5, 8) |
|---|---|---|
| Batch 1 | 4,549 | +0.0029 |
| Batch 2 | 4,440 | +0.0018 |
| Batch 3 | 4,501 | −0.0002 |
| Batch 4 + test set | 5,164 | +0.0002 |
| All 400 tuning stays | 18,654 | about +0.0012 |

The mean is less than a quarter of the 0.005 limit, and even the costliest 100-stay subset is under 0.003. The candidate passes my acceptance lines, and the cost model predicted batch 1 almost exactly. I will keep this edit. Pushing further would need my last Python run with no way to revert, for gains of only a few percent of the gap.

Next I measure the effects with dose_edit on the 75 test stays, where I have target values from the records.

**Prediction:**

| Edit | Original | Predicted for mk2(2.5, 8) |
|---|---|---|
| A x2 | −2.65 | ≈ −3.5 |
| A x1/2 | +1.27 | ≈ +1.5 |
| B x2 | +0.75 | ≈ +1.4 |
| B x1/2 | −0.44 | ≈ −1.1 |

**How I will read it:** values near these would confirm that the effects scale linearly with the gains. Clearly smaller shifts would mean the effects do not scale down linearly with the edit size.

```forward
{"type": "dose_edit", "model": "both", "split": "tuning", "stays": [2020000000,2020000008,2020000013,2020000018,2020000019,2020000023,2020000032,2020000038,2020000039,2020000040,2020000041,2020000048,2020000052,2020000060,2020000063,2020000069,2020000071,2020000074,2020000076,2020000081,2020000084,2020000086,2020000089,2020000091,2020000094,2020000102,2020000111,2020000112,2020000117,2020000120,2020000124,2020000126,2020000130,2020000132,2020000154,2020000156,2020000158,2020000165,2020000171,2020000176,2020000181,2020000182,2020000183,2020000187,2020000188,2020000198,2020000201,2020000204,2020000211,2020000212,2020000214,2020000232,2020000249,2020000250,2020000252,2020000254,2020000264,2020000265,2020000267,2020000268,2020000274,2020000279,2020000281,2020000286,2020000288,2020000293,2020000298,2020000300,2020000304,2020000305,2020000308,2020000309,2020000317,2020000321,2020000342]}
```
Fresh training batch 5 costs +.0028 per reading at -2.7. Every factual measurement of the insulin-only edit is now in, scaled to -2.7 by gain² where it was measured at another gain:

| Group | Batch costs | Mean |
|---|---|---|
| 4 tuning batches | .0046, .0043, .0033, .0011 | .0033 |
| 4 earlier training batches | .0018, .0028, .0040, .0011 | .0024 |
| 6 fresh training batches | .0027, .0032, .0010, .0002, .0026, .0028 | .0021 |

All 14 batches together average about .0025, with a batch standard deviation of .0013. The tuning-versus-training gap is not significant (t ≈ 1.4), and the original model's NLL does not suggest overfitting. For an evaluation set of a few hundred unseen stays, the chance of exceeding .005 at -2.7 is therefore small: well under 1% on the pooled estimate, a few percent on the most pessimistic tuning-only one.

Raising the gain further would gain only a few percent on the insulin readouts for a comparable increase in risk, so I will keep -2.7.

With passes to spare, I will now measure the readouts of the final edit directly instead of extrapolating. I will run dose_edit with `ins_only(-2.7)` on all four tuning batches, whose true shifts on the reached rows I computed earlier: pooled insulin x2 -13.55, x1/2 +6.77. I expect pooled ratios of about .95 for x2 and .90 for x1/2, with NE unchanged from the original. Batch f3 first.

```forward
{"type": "dose_edit", "model": "both", "split": "tuning", "stays": [2020000000,2020000001,2020000002,2020000003,2020000004,2020000005,2020000006,2020000007,2020000008,2020000009,2020000010,2020000011,2020000012,2020000013,2020000014,2020000015,2020000016,2020000017,2020000018,2020000019,2020000020,2020000021,2020000022,2020000023,2020000024,2020000025,2020000026,2020000027,2020000028,2020000029,2020000030,2020000031,2020000032,2020000033,2020000034,2020000035,2020000036,2020000037,2020000038,2020000039,2020000040,2020000041,2020000042,2020000043,2020000044,2020000045,2020000046,2020000047,2020000048,2020000049,2020000050,2020000051,2020000052,2020000053,2020000054,2020000055,2020000056,2020000057,2020000058,2020000059,2020000060,2020000061,2020000062,2020000063,2020000064,2020000065,2020000066,2020000067,2020000068,2020000069,2020000070,2020000071,2020000072,2020000073,2020000074,2020000075,2020000076,2020000077,2020000078,2020000079,2020000080,2020000081,2020000082,2020000083,2020000084,2020000085,2020000086,2020000087,2020000088,2020000089,2020000090,2020000091,2020000092,2020000093,2020000094,2020000095,2020000096,2020000097,2020000098,2020000099]}
```
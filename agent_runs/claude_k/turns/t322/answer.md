Pass 16 shows the cost now scales as expected, but leaves less safety margin than I wanted:

| Readings | NLL change at κ = 0.5 |
|---|---|
| All values | +0.0029 (1.4852 → 1.4881) |
| map_arterial | +0.0047 |
| map_cuff | +0.0062 |
| Glucose | unchanged |

- **Scaling:** κ = 1.5 cost 0.034, and a third of the gain gives 0.0029. That is about 0.087 of the cost, close to the 1/9 that pure κ² scaling predicts. So the cost is reader noise and scales with the square of the gain.
- **Margin:** 0.0029 is under the 0.005 limit, but its sampling uncertainty over 100 stays is about ±0.0013. On unseen stays it could plausibly come near the limit.
- **Likely final choice:** κ about 0.35–0.4, for an expected cost of about 0.0015–0.002.

Before choosing, I want to check that the rollout response scales linearly with κ at this setting, about +0.37 on norepinephrine ×2 and −0.06 on ×½. The rollout feeds the shifted readings back, so this needs checking. I will use the same 100 tuning stays with both models.

| Outcome | Conclusion |
|---|---|
| ×2 ≈ +0.7, ×½ ≈ −0.11, insulin unchanged | Linear scaling holds; a final κ follows directly |
| Response much smaller than linear scaling predicts | The reader is not worth any factual cost; I would remove it entirely |

```forward
{"type": "dose_edit", "model": "both", "split": "tuning", "stays": [2020000000,2020000008,2020000012,2020000013,2020000018,2020000019,2020000023,2020000031,2020000032,2020000034,2020000038,2020000039,2020000040,2020000041,2020000043,2020000048,2020000052,2020000053,2020000056,2020000058,2020000060,2020000063,2020000067,2020000069,2020000071,2020000074,2020000076,2020000079,2020000081,2020000084,2020000086,2020000089,2020000091,2020000093,2020000094,2020000096,2020000102,2020000111,2020000112,2020000117,2020000120,2020000124,2020000125,2020000126,2020000128,2020000130,2020000132,2020000154,2020000156,2020000158,2020000165,2020000171,2020000176,2020000181,2020000182,2020000183,2020000187,2020000188,2020000198,2020000201,2020000204,2020000211,2020000212,2020000214,2020000232,2020000249,2020000250,2020000252,2020000254,2020000264,2020000265,2020000267,2020000268,2020000274,2020000279,2020000281,2020000286,2020000288,2020000293,2020000298,2020000300,2020000304,2020000305,2020000308,2020000309,2020000317,2020000321,2020000342,2020000348,2020000352,2020000353,2020000361,2020000366,2020000370,2020000381,2020000382,2020000384,2020000391,2020000396,2020000398]}
```
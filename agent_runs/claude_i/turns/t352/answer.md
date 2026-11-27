The early-hour hypothesis holds. 98% of the factual damage is in readings from hours −1 to 4, almost all of it at hours ≤ 1. There the edit shifts finger glucose by about −14 and MAP by about +3. From hour 4 on, the NLL change is near zero.

If I removed that early misfire, the total NLL change would drop from about +0.037 to about +0.001, which is inside the 0.005 limit.

The full-strength edit's dose response (pass 5) is mixed:

| Edit | Edited model | True (target) |
|---|---|---|
| insulin ×2 | −7.2 | −12.5 |
| insulin ×½ | +5.6 | +6.25 |
| norepinephrine ×2 | +0.29 | +4.33 |

So the insulin channel partly works and the norepinephrine channel does nothing.

The misfire most likely happens at the very first bundle, the admission at −2h, where the model's state is still zero. Two candidate causes:
- the dose-deviation units in z_post fire on that unusual starting context;
- the attention-weight edit picks up the first bundle's slot tokens, which are just the slot embeddings because the state is zero.

To tell them apart, I'll fetch per-bundle states for both models on 20 stays, including the 5 worst. If z dims 18/28 are large at the first bundle, the cause is z_post. If they are near 0 but the state already differs a lot after the first bundle, the cause is the attention edit.

```forward
{"type": "states", "model": "both", "split": "tuning", "stays": [2020000031, 2020000032, 2020000094, 2020000041, 2020000264, 2020000000, 2020000001, 2020000003, 2020000007, 2020000008, 2020000010, 2020000012, 2020000013, 2020000015, 2020000016, 2020000018, 2020000019, 2020000022, 2020000023, 2020000024], "bundles": "all"}
```
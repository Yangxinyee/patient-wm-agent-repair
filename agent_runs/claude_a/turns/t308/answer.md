The bootstrap gives the tail risk I needed. For a 100-stay evaluation set, the 99.5th-percentile factual cost is:

| θₕ | 99.5th percentile | Chance of exceeding 0.005 |
|---|---|---|
| 0 | 0.0037 | about 0 |
| 0.35 | 0.0039 | about 0 |
| 0.5 | 0.0044 | 0.05% |
| 0.7 | 0.0052 | 0.8% |

For 400-stay sets, all settings stay at or below 0.0034. So θₕ = 0.7 carried a real risk of failing the factual limit on a small evaluation set, and the run stepped down to θₕ = 0.5, with θ₂ = 0.83.

That setting is saved. Offline on all 400 tuning stays it gives:

| Readout | Saved edit | Truth |
|---|---|---|
| insulin ×2 | −13.76 | −13.41 |
| insulin ×½ | +6.67 | +6.71 |
| norepinephrine ×2 | +4.39 | +4.47 |
| norepinephrine ×½ | −1.21 | −2.24 |

The average factual cost over 1,000 stays is +0.00095.

Now I verify the saved edit with the real tool on all 400 tuning stays, in four batches of 100. I can pool the batch summaries by hand, so this costs no Python runs. This first batch is the 100 tuning stays with the lowest ids not in sel1.

What I expect from the pooled tool readouts, given the replica runs about 2.5% strong on insulin:
- insulin about −13.4 and +6.5;
- norepinephrine about +4.4 and −1.2.

A pooled norepinephrine ×2 far from +4.4 would mean the tool and replica disagree on the gated norepinephrine readout.

```forward
{"type": "dose_edit", "model": "edited", "split": "tuning", "stays": [2020000002, 2020000004, 2020000005, 2020000006, 2020000007, 2020000008, 2020000009, 2020000011, 2020000014, 2020000015, 2020000016, 2020000017, 2020000018, 2020000019, 2020000020, 2020000021, 2020000022, 2020000023, 2020000025, 2020000026, 2020000028, 2020000029, 2020000030, 2020000032, 2020000033, 2020000034, 2020000035, 2020000036, 2020000037, 2020000039, 2020000040, 2020000042, 2020000043, 2020000044, 2020000045, 2020000048, 2020000049, 2020000050, 2020000052, 2020000053, 2020000054, 2020000055, 2020000056, 2020000057, 2020000058, 2020000059, 2020000060, 2020000061, 2020000063, 2020000064, 2020000065, 2020000066, 2020000067, 2020000068, 2020000071, 2020000072, 2020000074, 2020000075, 2020000076, 2020000077, 2020000078, 2020000081, 2020000082, 2020000083, 2020000085, 2020000086, 2020000087, 2020000088, 2020000089, 2020000090, 2020000091, 2020000093, 2020000094, 2020000095, 2020000096, 2020000097, 2020000098, 2020000100, 2020000101, 2020000102, 2020000103, 2020000104, 2020000105, 2020000106, 2020000107, 2020000108, 2020000110, 2020000111, 2020000112, 2020000113, 2020000114, 2020000115, 2020000118, 2020000119, 2020000121, 2020000122, 2020000123, 2020000126, 2020000127, 2020000128]}
```
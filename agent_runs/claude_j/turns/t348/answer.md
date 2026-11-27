Run 294 cleared up three things:

- **What is saved:** the current edit is mk2(12,12). It was saved at minute 14.5, so forward passes 10 and 11 tested it.
- **Factual cost (fwd 11, 75 tuning stays):** value NLL went from 1.50009 to 1.51706, a rise of +0.017. That is far over the 0.005 limit.
  - Almost all of the cost is in the two G events: +0.035 and +0.053.
  - The two M events barely moved: +0.003 and −0.006.
  - So the drug-A part of the amplification is what is expensive. The drug-B part is cheap.
- **Effects (fwd 10):** only the pooled mean shift over all four edits is readable. It moved from −0.29 to −1.47, against −0.52 for uniform x4. The per-edit numbers are in summary.json.

The tool is deterministic for a fixed set of stays: the original model's rows were identical in fwd 8 and fwd 10. Re-running the same dose_edit request should therefore reproduce fwd 10 and show the per-edit breakdown, without spending one of my 6 remaining Python runs.

**Hypothesis:** the A-edit effect grows roughly linearly in kA, and the B-edit effect grows only weakly in kB.
- **Confirmed** if A x2 lands far beyond the original −2.65, around −8 to −12, with A x1/2 about half of it and opposite in sign. Then a much smaller kA, about 4–5, should fit the cost limit if cost grows roughly as (kA−1)², while kB can stay high.
- **Refuted** if A x2 barely moved. That would mean the factual cost buys almost no effect, and I would fall back to uniform x4.

```forward
{"type": "dose_edit", "model": "both", "split": "tuning", "stays": [2020000000,2020000008,2020000013,2020000018,2020000019,2020000023,2020000032,2020000038,2020000039,2020000040,2020000041,2020000048,2020000052,2020000060,2020000063,2020000069,2020000071,2020000074,2020000076,2020000081,2020000084,2020000086,2020000089,2020000091,2020000094,2020000102,2020000111,2020000112,2020000117,2020000120,2020000124,2020000126,2020000130,2020000132,2020000154,2020000156,2020000158,2020000165,2020000171,2020000176,2020000181,2020000182,2020000183,2020000187,2020000188,2020000198,2020000201,2020000204,2020000211,2020000212,2020000214,2020000232,2020000249,2020000250,2020000252,2020000254,2020000264,2020000265,2020000267,2020000268,2020000274,2020000279,2020000281,2020000286,2020000288,2020000293,2020000298,2020000300,2020000304,2020000305,2020000308,2020000309,2020000317,2020000321,2020000342]}
```
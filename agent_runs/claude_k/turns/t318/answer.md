Pass 14 shows the norepinephrine reader works in the rollout, but its response is weaker than needed and lopsided:

| Edit | Original | Edited (κ = 1.5) | Reader's contribution | Rough target |
|---|---|---|---|---|
| Norepinephrine ×2 | +0.34 | +1.46 | +1.12 | +4.6 |
| Norepinephrine ×½ | −0.06 | −0.23 | −0.17 | −2.3 |
| Insulin ×2 | −2.15 | −2.12 | — | −13.8 |
| Insulin ×½ | +1.23 | +1.25 | — | +6.9 |

- **Gate:** the insulin results are unchanged, so it does not leak into the glucose codes.
- **Asymmetry:** if the reader responded linearly to the rate change, the ×½ contribution would be about −0.56. It is only −0.17.
- **Source of the asymmetry:** the original model shows the same lopsidedness (+0.34 against −0.06). So it comes from how the model's state encodes the norepinephrine rate: raising the rate moves the state much more than lowering it. It is not a reader artefact, and one gain cannot fix both edits.
- **Gain if fit alone:** fitting both edits by least squares on the reader's scale would need κ ≈ 6 (factor 4). That would give about +4.8 for ×2 and −0.7 for ×½.

What really limits κ is the factual fit. The cost of the reader's noise grows roughly with κ², so I must measure the cost at the current κ before choosing a larger one. Next is a factual pass on the same 100 tuning stays with both models.

| Factual cost at κ = 1.5 | Implication |
|---|---|
| ≤ about 0.0003 nats | κ can go up to about 5–6 within the 0.005 limit, with margin |
| ≥ about 0.002 nats | κ must stay near 2–2.5, and norepinephrine gets only part of the way to its target |

```forward
{"type": "factual", "model": "both", "split": "tuning", "stays": [2020000000,2020000008,2020000012,2020000013,2020000018,2020000019,2020000023,2020000031,2020000032,2020000034,2020000038,2020000039,2020000040,2020000041,2020000043,2020000048,2020000052,2020000053,2020000056,2020000058,2020000060,2020000063,2020000067,2020000069,2020000071,2020000074,2020000076,2020000079,2020000081,2020000084,2020000086,2020000089,2020000091,2020000093,2020000094,2020000096,2020000102,2020000111,2020000112,2020000117,2020000120,2020000124,2020000125,2020000126,2020000128,2020000130,2020000132,2020000154,2020000156,2020000158,2020000165,2020000171,2020000176,2020000181,2020000182,2020000183,2020000187,2020000188,2020000198,2020000201,2020000204,2020000211,2020000212,2020000214,2020000232,2020000249,2020000250,2020000252,2020000254,2020000264,2020000265,2020000267,2020000268,2020000274,2020000279,2020000281,2020000286,2020000288,2020000293,2020000298,2020000300,2020000304,2020000305,2020000308,2020000309,2020000317,2020000321,2020000342,2020000348,2020000352,2020000353,2020000361,2020000366,2020000370,2020000381,2020000382,2020000384,2020000391,2020000396,2020000398]}
```
Run 298 split the quiet delta and chose the gains, but the factual model narrowed the room more than I hoped.

**What the factual model says.** Per output, the estimated effect of the mean shift on fit is slightly favourable in the designed direction, not the flipped one. So my guess that flipping would improve the factual fit is not supported. Flipping costs factual fit both linearly and through the quadratic term; at −2 for both gains the predicted cost is 0.0085, which would fail.

**What was installed.** The largest flip within the 0.0025 budget is kI = −1.0 for insulin and kN = −0.75 for NE, with a predicted cost of +0.0024. If the dose response stays linear, that should give about:

| Readout | Expected | Original |
|---|---|---|
| Insulin x2 | −3.1 | −2.0 |
| Insulin x1/2 | +1.66 | +1.16 |
| NE x2 | +0.49 | +0.37 |
| NE x1/2 | −0.11 | −0.06 |

That is a modest improvement. Since it is close to the factual limit, I need to check it before deciding.

**Next: factual "both" pass.**
- **Confirmed:** NLL change of about +0.002 or less, with most of it on glucose_finger. Then I run the dose_edit.
- **Refuted:** if it is above about +0.003, the margin for unseen stays is too thin. I will then use run 299 to scale down or revert, keeping run 300 for a final revert if needed.

```forward
{"type": "factual", "model": "both", "split": "tuning", "stays": [2020000000, 2020000008, 2020000013, 2020000018, 2020000019, 2020000023, 2020000032, 2020000038, 2020000039, 2020000040, 2020000041, 2020000048, 2020000052, 2020000060, 2020000063, 2020000069, 2020000071, 2020000074, 2020000076, 2020000081, 2020000084, 2020000086, 2020000089, 2020000091, 2020000094, 2020000102, 2020000111, 2020000112, 2020000117, 2020000120, 2020000124, 2020000126, 2020000130, 2020000132, 2020000154, 2020000156, 2020000158, 2020000165, 2020000171, 2020000176, 2020000181, 2020000182, 2020000183, 2020000187, 2020000188, 2020000198, 2020000201, 2020000204, 2020000211, 2020000212, 2020000214, 2020000232, 2020000249, 2020000250, 2020000252, 2020000254, 2020000264, 2020000265, 2020000267, 2020000268, 2020000001, 2020000003, 2020000010, 2020000016, 2020000022, 2020000024, 2020000025, 2020000026, 2020000027, 2020000033, 2020000035, 2020000042, 2020000045, 2020000046, 2020000047, 2020000049, 2020000051, 2020000054, 2020000057, 2020000059, 2020000007, 2020000012, 2020000015, 2020000031, 2020000034, 2020000043, 2020000050, 2020000053, 2020000056, 2020000058, 2020000066, 2020000067, 2020000079, 2020000093, 2020000095, 2020000096, 2020000097, 2020000110, 2020000125, 2020000127]}
```
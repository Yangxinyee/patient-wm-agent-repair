The readout check supports the v5 targets. Dividing each readout dot product by the normalisation scale (`sig_m`, about 58.6) gives the immediate shift in the predicted mean per unit rate:

| Target | Finger glucose | Arterial MAP | Cross-talk |
|---|---|---|---|
| v4 insulin | −0.30 mg/dL | — | — |
| v5 insulin | −0.94 mg/dL | — | ≈0 on MAP |
| v4 NE | — | +0.13 mmHg | −0.54 mg/dL on glucose |
| v5 NE | — | +0.99 mmHg | −0.05 mg/dL on glucose |

So the v5 insulin target is about 3× stronger than v4's. The v5 NE target is about 7× stronger, and it removes v4's large NE-to-glucose cross-talk. That explains the weak v6 NE result.

v7(1,1), which combines the leak-free readouts with the v5 targets, is saved. Scaling v6's measured effects, I expect the edit's added effect to be about −3.4 mg/dL for insulin x2 and about +0.6 mmHg for NE x2, with x1/2 about −½ of each.

Next I run v7(1,1) through the same dose_edit pass on sel100. This gives a clean gain-1 calibration point before I move to the large target gains, where nonlinearity may appear. An added effect near those predictions, with roughly linear x2/x1/2 behaviour, confirms that the v5 targets act as designed.

```forward
{"type": "dose_edit", "model": "both", "split": "tuning", "stays": [2020000001, 2020000002, 2020000005, 2020000006, 2020000009, 2020000012, 2020000014, 2020000024, 2020000025, 2020000026, 2020000029, 2020000035, 2020000042, 2020000043, 2020000049, 2020000051, 2020000057, 2020000064, 2020000066, 2020000073, 2020000075, 2020000076, 2020000079, 2020000085, 2020000088, 2020000089, 2020000090, 2020000091, 2020000097, 2020000099, 2020000101, 2020000103, 2020000109, 2020000111, 2020000112, 2020000114, 2020000117, 2020000119, 2020000120, 2020000126, 2020000131, 2020000133, 2020000135, 2020000138, 2020000141, 2020000144, 2020000145, 2020000154, 2020000158, 2020000159, 2020000164, 2020000169, 2020000170, 2020000171, 2020000174, 2020000177, 2020000182, 2020000188, 2020000197, 2020000200, 2020000215, 2020000220, 2020000223, 2020000224, 2020000227, 2020000231, 2020000233, 2020000236, 2020000238, 2020000240, 2020000247, 2020000249, 2020000253, 2020000257, 2020000259, 2020000262, 2020000274, 2020000277, 2020000280, 2020000300, 2020000301, 2020000304, 2020000310, 2020000311, 2020000317, 2020000319, 2020000321, 2020000322, 2020000325, 2020000328, 2020000334, 2020000335, 2020000355, 2020000361, 2020000362, 2020000365, 2020000369, 2020000377, 2020000386, 2020000387]}
```
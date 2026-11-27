The per-batch comparison confirms the calibration. Tool and truth track each other batch by batch: for insulin ×2, −12.50/−14.79/−13.29/−13.71 against −12.32/−14.61/−13.99/−13.25. Pooled over all 400 tuning stays, the tool-minus-truth gaps are:

| Readout | Gap | Standard error |
|---|---|---|
| insulin ×2 | −0.11 | 0.28 |
| insulin ×½ | −0.22 | 0.20 |
| norepinephrine ×2 | −0.12 | 0.27 |
| norepinephrine ×½ | +0.95 | 0.17 |

The first three gaps are within about one standard error, so they are calibrated as well as 400 stays allow. The norepinephrine ×½ gap is real and deliberate, because a fuller repair would cost too much factual fit.

I made a small nudge of θ₂ from 0.83 to 0.868 to close the norepinephrine ×2 gap, and saved it. The offline factual cost on the 400 tuning stays is now +0.0005.

What remains is the hard constraint. I will check the final edit's factual fit with the real tool on all 400 tuning stays, in four "both" passes. I start with sel1, the most expensive subset. There the θₕ = 0.7 version cost +0.0036; with θₕ = 0.5 I expect about +0.0025 to +0.003. Over all 400 stays I expect about +0.0005.

What would confirm or refute the edit:
- **Confirm:** results matching those expectations.
- **Refute:** a pooled tool cost clearly above +0.002 would mean I need to lower θₕ further before finishing.

```forward
{"type": "factual", "model": "both", "split": "tuning", "stays": [2020000000, 2020000001, 2020000003, 2020000010, 2020000012, 2020000013, 2020000024, 2020000027, 2020000031, 2020000038, 2020000041, 2020000046, 2020000047, 2020000051, 2020000062, 2020000069, 2020000070, 2020000073, 2020000079, 2020000080, 2020000084, 2020000092, 2020000099, 2020000109, 2020000116, 2020000117, 2020000120, 2020000124, 2020000125, 2020000137, 2020000138, 2020000142, 2020000146, 2020000147, 2020000150, 2020000153, 2020000154, 2020000158, 2020000159, 2020000160, 2020000167, 2020000176, 2020000177, 2020000182, 2020000186, 2020000188, 2020000192, 2020000195, 2020000196, 2020000199, 2020000200, 2020000209, 2020000211, 2020000216, 2020000221, 2020000223, 2020000232, 2020000240, 2020000242, 2020000245, 2020000248, 2020000249, 2020000250, 2020000252, 2020000256, 2020000259, 2020000265, 2020000268, 2020000282, 2020000288, 2020000292, 2020000295, 2020000296, 2020000300, 2020000304, 2020000307, 2020000318, 2020000320, 2020000327, 2020000330, 2020000331, 2020000335, 2020000338, 2020000339, 2020000340, 2020000345, 2020000349, 2020000352, 2020000358, 2020000359, 2020000362, 2020000363, 2020000364, 2020000367, 2020000369, 2020000382, 2020000386, 2020000387, 2020000396, 2020000398]}
```
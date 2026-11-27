Pooling passes 7 to 10 again reproduces the earlier numbers for the edit before the last th2 tweak (th2 = 0.83):

| Edit | Model readout | Truth (400 tuning stays) |
|---|---|---|
| Insulin x2 | −13.52 | −13.42 |
| Insulin x1/2 | 6.49 | 6.71 |
| Norepinephrine x2 | 4.35 | 4.47 |
| Norepinephrine x1/2 | −1.29 | −2.24 |

Passes 7 to 10 used the same four 100-stay batches as the factual checks: pass 10 = B11, 7 = B12, 8 = B13, 9 = B14. The final edit's tensors are backed up.

Next I run the evaluation's rollout on the actual submission (hash 5c2b8ab571, th2 = 0.868) on each batch. I use "both" so the original model's readouts come with it. I will pool the four batches by stay-hours by hand.

What would confirm that the saved edit is the intended one:
- Insulin readouts essentially unchanged: about −13.5 and +6.5.
- Norepinephrine x2 about 0.1 higher than before, near 4.45. On B11 that means about 4.4 to 4.5, against 4.37 in pass 10.
- Norepinephrine x1/2 about −1.3.
- The original model's norepinephrine readouts near zero, as found earlier.

A clearly different pattern would mean the submitted weights are not the configuration I think they are.

```forward
{"type": "dose_edit", "model": "both", "split": "tuning", "stays": [2020000000, 2020000001, 2020000003, 2020000010, 2020000012, 2020000013, 2020000024, 2020000027, 2020000031, 2020000038, 2020000041, 2020000046, 2020000047, 2020000051, 2020000062, 2020000069, 2020000070, 2020000073, 2020000079, 2020000080, 2020000084, 2020000092, 2020000099, 2020000109, 2020000116, 2020000117, 2020000120, 2020000124, 2020000125, 2020000137, 2020000138, 2020000142, 2020000146, 2020000147, 2020000150, 2020000153, 2020000154, 2020000158, 2020000159, 2020000160, 2020000167, 2020000176, 2020000177, 2020000182, 2020000186, 2020000188, 2020000192, 2020000195, 2020000196, 2020000199, 2020000200, 2020000209, 2020000211, 2020000216, 2020000221, 2020000223, 2020000232, 2020000240, 2020000242, 2020000245, 2020000248, 2020000249, 2020000250, 2020000252, 2020000256, 2020000259, 2020000265, 2020000268, 2020000282, 2020000288, 2020000292, 2020000295, 2020000296, 2020000300, 2020000304, 2020000307, 2020000318, 2020000320, 2020000327, 2020000330, 2020000331, 2020000335, 2020000338, 2020000339, 2020000340, 2020000345, 2020000349, 2020000352, 2020000358, 2020000359, 2020000362, 2020000363, 2020000364, 2020000367, 2020000369, 2020000382, 2020000386, 2020000387, 2020000396, 2020000398]}
```
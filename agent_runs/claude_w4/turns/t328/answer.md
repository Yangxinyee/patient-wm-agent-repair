Lowering c made things worse. v9's value NLL rose +0.234, versus +0.173 for v7 at s = 0.3:

| Reading | v7 (s = 0.3) | v9 (c = 0.4) |
|---|---|---|
| Finger glucose | +0.038 | +0.145 |
| Lab glucose | +0.133 | +0.497 |
| Arterial MAP | +0.30 | +0.26 |
| Cuff MAP | +0.23 | +0.23 |

The insulin gain was even lower in v9 (1.0 against 1.455).

What changed is that b_in now amplifies whatever activity lies along the channel direction 15× more. Such activity is only the intended signal at the drug bundles I sampled; elsewhere it is unwanted. So the damage most likely comes from natural or out-of-sample activity along the channel at some bundles I never sampled. The amplified edit maps it into large injections.

The damage also looks saturated: median NLL change is about 0.001 while the mean is large, so a few bundles cause most of the harm. That would explain why cutting the gain barely helped at c = 6.

To find those bundles directly, I ask for the model state after every bundle, for both models, on the same 100 stays. In the next run I will:
1. find, per stay, where the edited state first diverges sharply from the original;
2. check whether those bundles share a feature, such as the first bundle of a stay, end-of-infusion bundles, drug bundles, or a particular hour.

If they do, I can block that path cleanly. If the divergence is spread over all drug bundles, the intended injection itself is the problem.

```forward
{"type": "states", "model": "both", "split": "tuning", "stays": [2020000001, 2020000002, 2020000005, 2020000006, 2020000009, 2020000012, 2020000014, 2020000024, 2020000025, 2020000026, 2020000029, 2020000035, 2020000042, 2020000043, 2020000049, 2020000051, 2020000057, 2020000064, 2020000066, 2020000073, 2020000075, 2020000076, 2020000079, 2020000085, 2020000088, 2020000089, 2020000090, 2020000091, 2020000097, 2020000099, 2020000101, 2020000103, 2020000109, 2020000111, 2020000112, 2020000114, 2020000117, 2020000119, 2020000120, 2020000126, 2020000131, 2020000133, 2020000135, 2020000138, 2020000141, 2020000144, 2020000145, 2020000154, 2020000158, 2020000159, 2020000164, 2020000169, 2020000170, 2020000171, 2020000174, 2020000177, 2020000182, 2020000188, 2020000197, 2020000200, 2020000215, 2020000220, 2020000223, 2020000224, 2020000227, 2020000231, 2020000233, 2020000236, 2020000238, 2020000240, 2020000247, 2020000249, 2020000253, 2020000257, 2020000259, 2020000262, 2020000274, 2020000277, 2020000280, 2020000300, 2020000301, 2020000304, 2020000310, 2020000311, 2020000317, 2020000319, 2020000321, 2020000322, 2020000325, 2020000328, 2020000334, 2020000335, 2020000355, 2020000361, 2020000362, 2020000365, 2020000369, 2020000377, 2020000386, 2020000387]}
```
Of the 100 sel100 tuning stays, 45 have insulin, norepinephrine and an arterial line. The factual predicted means for all 100 are already in fwd_2.

I will now spend one forward pass to get the original model's states after every bundle for 12 of these stays. That gives me h, out and the z posterior at each step. With them I can check a local reimplementation one step at a time (encoders and z_post first, then prelude, blocks and coda) without errors building up. If my reimplemented z_post matches the returned z means to about 1e-4, the embedding and encoders are right.

```forward
{"type": "states", "model": "original", "split": "tuning", "stays": [2020000003, 2020000006, 2020000015, 2020000033, 2020000034, 2020000041, 2020000055, 2020000075, 2020000093, 2020000105, 2020000110, 2020000112], "bundles": "all"}
```
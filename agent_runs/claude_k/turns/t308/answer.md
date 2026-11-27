The last run saved the input-level amplification edit at k=6. It multiplies the z_dose and z_rate columns of `embed.numeric` by 6 and spreads the rate/dose bin embeddings around bin 8 by 6. It also printed the 100 tuning stays used for dose-edit calibration (the same set as fwd11).

**What I conclude so far:**
- The earlier routes are dead ends:
  - writing a z-channel: its signal was buried in the prelude layer norm;
  - amplifying the action-encoder output: insulin's u saturated and the factual fit distorted;
  - a probe in the value head: factual cost too high.
- The remaining lever is the model's own dose pathway, made louder at the input. More dose signal may then survive the drug-bundle step and reach the glucose and MAP predictions.

**What I do next:** a dose_edit pass with "both" on these 100 tuning stays. It measures how much k=6 changes the four predicted shifts (insulin x2 and x1/2 on glucose, norepinephrine x2 and x1/2 on MAP) compared with the original model, using the same random numbers.

**What would confirm or refute the approach:**
- Confirm: the edited shifts become clearly larger than the original's (about -2.1 for insulin x2) in a dose-ordered way (x2 and x1/2 move in opposite directions). Then k can be calibrated against the true effects once I re-read them from the notebook head.
- Refute: the shifts are barely changed, or random-looking and not ordered by dose. That would mean the bottleneck is downstream of the input embeddings and this approach should be dropped.
- Note that the shared embedding amplifies both drugs at once. If the true effects differ by drug, I will still need a drug-specific adjustment.

```forward
{"type": "dose_edit", "model": "both", "split": "tuning", "stays": [2020000000, 2020000008, 2020000012, 2020000013, 2020000018, 2020000019, 2020000023, 2020000031, 2020000032, 2020000034, 2020000038, 2020000039, 2020000040, 2020000041, 2020000043, 2020000048, 2020000052, 2020000053, 2020000056, 2020000058, 2020000060, 2020000063, 2020000067, 2020000069, 2020000071, 2020000074, 2020000076, 2020000079, 2020000081, 2020000084, 2020000086, 2020000089, 2020000091, 2020000093, 2020000094, 2020000096, 2020000102, 2020000111, 2020000112, 2020000117, 2020000120, 2020000124, 2020000125, 2020000126, 2020000128, 2020000130, 2020000132, 2020000154, 2020000156, 2020000158, 2020000165, 2020000171, 2020000176, 2020000181, 2020000182, 2020000183, 2020000187, 2020000188, 2020000198, 2020000201, 2020000204, 2020000211, 2020000212, 2020000214, 2020000232, 2020000249, 2020000250, 2020000252, 2020000254, 2020000264, 2020000265, 2020000267, 2020000268, 2020000274, 2020000279, 2020000281, 2020000286, 2020000288, 2020000293, 2020000298, 2020000300, 2020000304, 2020000305, 2020000308, 2020000309, 2020000317, 2020000321, 2020000342, 2020000348, 2020000352, 2020000353, 2020000361, 2020000366, 2020000370, 2020000381, 2020000382, 2020000384, 2020000391, 2020000396, 2020000398]}
```
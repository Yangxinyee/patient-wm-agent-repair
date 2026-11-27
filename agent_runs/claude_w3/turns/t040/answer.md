On the exact sel100 rows, assuming linearity, the expected true readouts are:

| edit | truth estimate | se | original model |
|---|---|---|---|
| insulin ×2 | −14.1 | 1.2 | −0.39 |
| insulin ×½ | +7.1 | 0.6 | +0.15 |
| norepinephrine ×2 | +3.8 | 0.27 | −0.66 |
| norepinephrine ×½ | −1.9 | 0.13 | −0.24 |

The model is off by an order of magnitude on insulin and has the wrong sign on norepinephrine ×2. One open caveat is that norepinephrine ×2 pushes many rates above the observed maximum of 1.0, so I will test for saturation within 0–1 later.

Now the editing part. I will record these targets, then request the model's states for the same 100 tuning stays. With the states I can rebuild the view, query and value head locally, and check them against the factual predicted means already in fwd_2. If that works, head edits can be scored for factual NLL exactly in the sandbox without spending forward passes. The states will also show whether recent doses can be decoded linearly from the state. If I can reproduce the predicted means closely, the local head is right. If not, I will learn which details (activation, type index, z sampling) I have wrong.

```forward
{"type": "states", "model": "original", "split": "tuning", "stays": [2020000003, 2020000006, 2020000007, 2020000015, 2020000021, 2020000030, 2020000033, 2020000034, 2020000036, 2020000038, 2020000041, 2020000054, 2020000055, 2020000066, 2020000075, 2020000085, 2020000093, 2020000105, 2020000110, 2020000112, 2020000113, 2020000115, 2020000123, 2020000128, 2020000130, 2020000140, 2020000141, 2020000146, 2020000150, 2020000158, 2020000159, 2020000164, 2020000170, 2020000177, 2020000179, 2020000185, 2020000187, 2020000200, 2020000203, 2020000209, 2020000211, 2020000217, 2020000219, 2020000221, 2020000242, 2020000243, 2020000244, 2020000245, 2020000248, 2020000255, 2020000258, 2020000261, 2020000263, 2020000273, 2020000281, 2020000285, 2020000288, 2020000290, 2020000293, 2020000294, 2020000299, 2020000302, 2020000307, 2020000313, 2020000321, 2020000325, 2020000331, 2020000341, 2020000350, 2020000351, 2020000352, 2020000356, 2020000357, 2020000370, 2020000371, 2020000374, 2020000382, 2020000386, 2020000388, 2020000395, 2020000399, 2020000304, 2020000306, 2020000233, 2020000186, 2020000103, 2020000016, 2020000009, 2020000121, 2020000065, 2020000025, 2020000282, 2020000022, 2020000014, 2020000229, 2020000334, 2020000116, 2020000212, 2020000165, 2020000337]}
```
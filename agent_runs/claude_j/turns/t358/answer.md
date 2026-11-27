Forward pass 17 is encouraging. On batch 1, the stay set where the larger edit cost +0.0100, the fallback mk2(2.2, 5) costs only +0.0018:

| Event | Change in NLL |
|---|---|
| GF | +0.0030 |
| GL | +0.0052 |
| MA | +0.0014 |
| MC | +0.0000 |

**What I conclude:**
- At about 30% of the edit's size, the cost fell to 18% of its value. So the cost grows faster than linearly with edit size.
- The M-event cost behaves almost quadratically, so the B channel is cheap when small.
- The G-event cost is closer to linear, so the A channel is harder to make cheap.
- A rough split of the cost between the two channels suggests a final candidate near mk2(2.8, 7). It should cost about 0.0026 on a population-like mix of stays, and move the effects a little further.

Before relying on that, I need to know how much the cost varies between stay sets. I will measure the current fallback on batches 2, 3 and 4 and on the test set. Then:
- Run 299 will set the final candidate, which I will validate the same way.
- Run 300 is a guaranteed revert to this fallback if the final fails.

This pass is batch 2. The cost model is confirmed if it lands near batch 1's +0.0018. If it is much higher, stay sets vary a lot, and I should keep the fallback or shrink further.

```forward
{"type": "factual", "model": "both", "split": "tuning", "stays": [2020000134,2020000135,2020000136,2020000137,2020000138,2020000139,2020000140,2020000141,2020000142,2020000143,2020000144,2020000145,2020000146,2020000147,2020000148,2020000149,2020000150,2020000151,2020000152,2020000153,2020000155,2020000157,2020000159,2020000160,2020000161,2020000162,2020000163,2020000164,2020000166,2020000167,2020000168,2020000169,2020000170,2020000172,2020000173,2020000174,2020000175,2020000177,2020000178,2020000179,2020000180,2020000184,2020000185,2020000186,2020000189,2020000190,2020000191,2020000192,2020000193,2020000194,2020000195,2020000196,2020000197,2020000199,2020000200,2020000202,2020000203,2020000205,2020000206,2020000207,2020000208,2020000209,2020000210,2020000213,2020000215,2020000216,2020000217,2020000218,2020000219,2020000220,2020000221,2020000222,2020000223,2020000224,2020000225,2020000226,2020000227,2020000228,2020000229,2020000230,2020000231,2020000233,2020000234,2020000235,2020000236,2020000237,2020000238,2020000239,2020000240,2020000241,2020000242,2020000243,2020000244,2020000245,2020000246,2020000247,2020000248,2020000251,2020000253,2020000255]}
```
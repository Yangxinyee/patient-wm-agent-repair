Block 100–199 gives the same picture: the c=1 edit costs +0.0072 nats per reading (1.62622 against 1.61898).
- **Cuff MAP** is again the dominant loss, +0.020 per reading. That is about 30 of the block's roughly 35 nats.
- **Glucose** adds a little: finger +0.005, lab +0.003.
- **Arterial MAP** actually improves, by −0.004.

Three independent sets now agree: +0.0090, +0.0073 and +0.0072. So the damage is stable and general, not driven by a few odd stays. Its main source is cuff MAP, consistently +0.014 to +0.020 per reading, while arterial MAP, the reading the norepinephrine effect was built for, is roughly neutral.

The open question for the final run is whether these cuff losses follow insulin events or norepinephrine events.
- **Norepinephrine:** the last run can keep insulin's strength fairly high and cut norepinephrine's.
- **Insulin:** that is cross-talk from the insulin effect onto MAP, and insulin's strength must come down.

**Next.** Block 200–299, keeping the same procedure so the final split between drugs uses all 400 tuning stays.

**What would confirm or refute.** Another +0.007 to +0.009 loss led by cuff MAP would confirm a uniform pattern that can be scaled safely. A much larger value would mean heavier tails than I think, and the final budget must shrink.

```forward
{"type": "factual", "model": "both", "split": "tuning", "stays": [2020000200,2020000201,2020000202,2020000203,2020000204,2020000205,2020000206,2020000207,2020000208,2020000209,2020000210,2020000211,2020000212,2020000213,2020000214,2020000215,2020000216,2020000217,2020000218,2020000219,2020000220,2020000221,2020000222,2020000223,2020000224,2020000225,2020000226,2020000227,2020000228,2020000229,2020000230,2020000231,2020000232,2020000233,2020000234,2020000235,2020000236,2020000237,2020000238,2020000239,2020000240,2020000241,2020000242,2020000243,2020000244,2020000245,2020000246,2020000247,2020000248,2020000249,2020000250,2020000251,2020000252,2020000253,2020000254,2020000255,2020000256,2020000257,2020000258,2020000259,2020000260,2020000261,2020000262,2020000263,2020000264,2020000265,2020000266,2020000267,2020000268,2020000269,2020000270,2020000271,2020000272,2020000273,2020000274,2020000275,2020000276,2020000277,2020000278,2020000279,2020000280,2020000281,2020000282,2020000283,2020000284,2020000285,2020000286,2020000287,2020000288,2020000289,2020000290,2020000291,2020000292,2020000293,2020000294,2020000295,2020000296,2020000297,2020000298,2020000299]}
```
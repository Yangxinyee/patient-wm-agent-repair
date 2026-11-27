The NE teacher-forced check is noisy on 365 pairs: +0.03 ± 0.80 mmHg per mcg/kg/min, against about +3.0 ± 0.1 in the training records with the same controls. So the model probably has little causal NE effect either.

This creates a real constraint. The model's predictions are calibrated on average because its glucose dynamics already absorb the systematic, tier-driven part of each dose. If I amplified the response to the whole dose, factual predictions for steady insulin patients would double-count that part. The bias could reach −10 to −20 mg/dL, which would break the 0.005-nat budget.

A safe edit must therefore act on the dose's deviation from what the context predicts, not on its level. On the data manifold only the deviation is informative. The model's posterior latent z, which sees both the context and the dose, is a natural place for that deviation to live.

Next I will request the original model's states on the same 100 tuning stays. I will check:
1. whether the z posterior at insulin bundles varies with within-tier dose noise and with the tier base;
2. how the h and out states change across drug bundles.

This will show whether a dose-deviation direction exists to amplify.

```forward
{"type": "states", "model": "original", "split": "tuning", "stays": [2020000000, 2020000001, 2020000002, 2020000003, 2020000004, 2020000005, 2020000006, 2020000007, 2020000008, 2020000009, 2020000010, 2020000011, 2020000012, 2020000013, 2020000014, 2020000015, 2020000016, 2020000017, 2020000018, 2020000019, 2020000020, 2020000021, 2020000022, 2020000023, 2020000024, 2020000025, 2020000026, 2020000027, 2020000028, 2020000029, 2020000030, 2020000031, 2020000032, 2020000033, 2020000034, 2020000035, 2020000036, 2020000037, 2020000038, 2020000039, 2020000040, 2020000041, 2020000042, 2020000043, 2020000044, 2020000045, 2020000046, 2020000047, 2020000048, 2020000049, 2020000050, 2020000051, 2020000052, 2020000053, 2020000054, 2020000055, 2020000056, 2020000057, 2020000058, 2020000059, 2020000060, 2020000061, 2020000062, 2020000063, 2020000064, 2020000065, 2020000066, 2020000067, 2020000068, 2020000069, 2020000070, 2020000071, 2020000072, 2020000073, 2020000074, 2020000075, 2020000076, 2020000077, 2020000078, 2020000079, 2020000080, 2020000081, 2020000082, 2020000083, 2020000084, 2020000085, 2020000086, 2020000087, 2020000088, 2020000089, 2020000090, 2020000091, 2020000092, 2020000093, 2020000094, 2020000095, 2020000096, 2020000097, 2020000098, 2020000099]}
```
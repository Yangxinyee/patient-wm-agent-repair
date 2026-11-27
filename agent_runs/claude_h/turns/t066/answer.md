The ML-free estimators settle it: without the variance-reduction step, neither association is significant.

| Drug | Implied ×2 readout | se | z |
|---|---|---|---|
| Norepinephrine, combined MAP | −0.22 | 0.15 | −1.45 |
| Norepinephrine, arterial MAP | −0.22 | 0.16 | −1.37 |
| Insulin, finger glucose | +0.59 | 0.60 | +1.00 |

The stronger boosting-based figures (z of −2.6 and +1.8) depend on modelling choices that I have shown can shift results by about 0.1–0.15. Taken together, the records establish:
- **Insulin:** no detectable effect on finger glucose. Lag 1 is about 0, the dose response is flat, and the threshold jumps are null. Any true ×2 effect is within about ±1.5 mg/dL.
- **Norepinephrine:** no reliable effect on MAP from dose changes. Any true ×2 effect is at most a few tenths of a mmHg.

Exact zeros are therefore my answer, and the dose-invariant edit delivers them. Before finishing I will repeat the factual NLL comparison on 100 fresh tuning stays. This confirms the near-zero NLL change (+0.00005 on the first set) is not specific to those stays. The plan holds if the difference again sits far inside the 0.005 limit; anything near 0.005 would force a rethink.

```forward
{"type": "factual", "model": "both", "split": "tuning", "stays": [2020000002, 2020000011, 2020000014, 2020000016, 2020000020, 2020000024, 2020000027, 2020000038, 2020000041, 2020000042, 2020000043, 2020000045, 2020000049, 2020000051, 2020000054, 2020000056, 2020000058, 2020000059, 2020000061, 2020000062, 2020000063, 2020000064, 2020000069, 2020000071, 2020000072, 2020000073, 2020000074, 2020000075, 2020000077, 2020000078, 2020000079, 2020000082, 2020000083, 2020000085, 2020000086, 2020000088, 2020000090, 2020000092, 2020000094, 2020000095, 2020000096, 2020000097, 2020000098, 2020000099, 2020000100, 2020000103, 2020000105, 2020000106, 2020000107, 2020000108, 2020000109, 2020000110, 2020000111, 2020000112, 2020000113, 2020000115, 2020000116, 2020000117, 2020000119, 2020000120, 2020000122, 2020000123, 2020000124, 2020000125, 2020000126, 2020000128, 2020000129, 2020000130, 2020000131, 2020000132, 2020000133, 2020000134, 2020000135, 2020000136, 2020000138, 2020000139, 2020000140, 2020000142, 2020000144, 2020000145, 2020000146, 2020000147, 2020000148, 2020000149, 2020000151, 2020000152, 2020000154, 2020000156, 2020000157, 2020000159, 2020000160, 2020000163, 2020000164, 2020000165, 2020000166, 2020000168, 2020000169, 2020000170, 2020000171, 2020000173]}
```
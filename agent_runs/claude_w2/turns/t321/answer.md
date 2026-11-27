The tool confirms the final edit on the 100 sel1 tuning stays, and its numbers match my local engine within about 0.1.

| Edit | Original | Edited | Local prediction | My record-based estimate |
|---|---|---|---|---|
| NE ×2 | +1.08 | +3.13 | 3.17 | ≈ +4.7 |
| NE ×½ | −0.73 | −2.30 | −2.22 | ≈ −2.35 |
| Insulin ×2 | −2.14 | −1.92 | — | ≈ −16.5 |
| Insulin ×½ | +1.26 | +1.98 | — | ≈ +8.3 |

- **Pooled insulin result:** over sel1 and sel2 (1,191 stay-hours), insulin ×2 is unchanged: original −2.81, edited −2.76.
- **NE:** the edit is a clear improvement, about 3×, now near my record-based estimates, while factual fit improved slightly on both sets.
- **Insulin:** still far too weak. Every encoder-side lever I tried leaves it unchanged, and the head-readout route costs too much factual fit for too little effect. So I am stopping work on insulin.

One cheap check remains before finishing. That is a third paired factual test on another 100 tuning stays; I listed the IDs by hand, assuming tuning IDs are consecutive. It makes sure the factual-fit result is not a quirk of two samples.
- **Confirms:** the edited value NLL is again within about ±0.003 of the original, as on sel1 (−0.0021) and sel2 (−0.0002).
- **Refutes:** a change approaching +0.005, which would make me back off to a milder scale.

```forward
{"type": "factual", "model": "both", "split": "tuning", "stays": [2020000156, 2020000157, 2020000158, 2020000159, 2020000161, 2020000163, 2020000164, 2020000166, 2020000167, 2020000168, 2020000169, 2020000172, 2020000173, 2020000174, 2020000176, 2020000177, 2020000179, 2020000180, 2020000181, 2020000182, 2020000183, 2020000184, 2020000185, 2020000186, 2020000187, 2020000188, 2020000189, 2020000190, 2020000191, 2020000192, 2020000193, 2020000194, 2020000195, 2020000196, 2020000197, 2020000198, 2020000199, 2020000200, 2020000201, 2020000203, 2020000204, 2020000205, 2020000206, 2020000207, 2020000208, 2020000210, 2020000211, 2020000212, 2020000213, 2020000214, 2020000215, 2020000216, 2020000217, 2020000218, 2020000219, 2020000220, 2020000221, 2020000222, 2020000224, 2020000225, 2020000226, 2020000227, 2020000228, 2020000229, 2020000230, 2020000231, 2020000232, 2020000233, 2020000235, 2020000236, 2020000237, 2020000238, 2020000239, 2020000240, 2020000241, 2020000242, 2020000243, 2020000244, 2020000245, 2020000247, 2020000248, 2020000249, 2020000250, 2020000251, 2020000252, 2020000253, 2020000254, 2020000255, 2020000256, 2020000257, 2020000259, 2020000261, 2020000262, 2020000263, 2020000264, 2020000265, 2020000266, 2020000268, 2020000269, 2020000270]}
```
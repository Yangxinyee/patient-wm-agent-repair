The first 100-stay batch looks right. On the tool:

| Readout | Batch A | 400-stay truth |
|---|---|---|
| insulin ×2 | −12.50 | −13.41 |
| insulin ×½ | +5.76 | +6.71 |
| norepinephrine ×2 | +4.80 | +4.47 |
| norepinephrine ×½ | −1.90 | −2.24 |

This batch has only 50 insulin stays and 24 norepinephrine stays, and its own truth will differ from the 400-stay figures. So I can't judge it alone. I'll compute the truth on exactly these rows once all four batches are in. The norepinephrine readout clearly works through the tool's GPU rollout, and the gating behaves the same as offline. The ×½ value is even a bit stronger than offline.

Next is batch B, the next 100 tuning stays not in sel1, ids 129 to 266. After batch C and sel1, one Python run will compare the tool's pooled readouts with the truth on the same rows.

What would confirm the edit:
- **Confirm:** the pooled ratios of tool to truth come out about 1.0 for insulin ×2 and norepinephrine ×2, about 0.95 for insulin ×½, and about 0.55–0.85 for norepinephrine ×½.
- **Refute:** large deviations from those ratios would call for a final recalibration.

```forward
{"type": "dose_edit", "model": "edited", "split": "tuning", "stays": [2020000129, 2020000130, 2020000131, 2020000132, 2020000133, 2020000134, 2020000135, 2020000136, 2020000139, 2020000140, 2020000141, 2020000143, 2020000144, 2020000145, 2020000148, 2020000149, 2020000151, 2020000152, 2020000155, 2020000156, 2020000157, 2020000161, 2020000162, 2020000163, 2020000164, 2020000165, 2020000166, 2020000168, 2020000169, 2020000170, 2020000171, 2020000172, 2020000173, 2020000174, 2020000175, 2020000178, 2020000179, 2020000180, 2020000181, 2020000183, 2020000184, 2020000185, 2020000187, 2020000189, 2020000190, 2020000191, 2020000193, 2020000194, 2020000197, 2020000198, 2020000201, 2020000202, 2020000203, 2020000204, 2020000205, 2020000206, 2020000207, 2020000208, 2020000210, 2020000212, 2020000213, 2020000214, 2020000215, 2020000217, 2020000218, 2020000219, 2020000220, 2020000222, 2020000224, 2020000225, 2020000226, 2020000227, 2020000228, 2020000229, 2020000230, 2020000231, 2020000233, 2020000234, 2020000235, 2020000236, 2020000237, 2020000238, 2020000239, 2020000241, 2020000243, 2020000244, 2020000246, 2020000247, 2020000251, 2020000253, 2020000254, 2020000255, 2020000257, 2020000258, 2020000260, 2020000261, 2020000262, 2020000263, 2020000264, 2020000266]}
```
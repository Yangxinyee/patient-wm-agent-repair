On the second set of 100 stays, NE now responds much more strongly to its dose, and insulin barely changes.

| Edit (sel2) | Original model | Edited model | My record-based estimate |
|---|---|---|---|
| NE ×2 on MAP | +1.26 | +4.08 | about +4.7 |
| NE ×½ on MAP | −0.95 | −2.62 | about −2.35 |
| Insulin ×2 on finger glucose | −4.18 | −4.46 | about −16.5 |
| Insulin ×½ on finger glucose | +2.23 | +2.63 | about +8.3 |

The NE gain therefore holds on stays that played no part in choosing the scale factors. NE uses only 13 stays here, so the size of the gain is noisy, but it is about 3× in both directions, consistent with the local scan on sel1.

Insulin stays far too weak. Every encoder-side change I tested leaves insulin ×2 near the original's value. Scaling the rate input by 10 also made no difference, which suggests the insulin pathway saturates or is not used further on. The head readout added only −1.5 for 0.044 nats. So I am not spending my last 3 Python runs on another insulin attempt that is very unlikely to work.

To report the final expected readouts on the larger set (84 insulin stays, 56 NE stays), I run the tool's dose_edit on sel1 with both models.
- **Confirms:** edited NE ×2 ≈ 3.2 and ×½ ≈ −2.2, with insulin ×2 ≈ −1.9 to −2.2, matching my local numbers.
- **Refutes:** a large gap from the local numbers would mean the local engine misled the choice.

```forward
{"type": "dose_edit", "model": "both", "split": "tuning", "stays": [2020000000, 2020000002, 2020000003, 2020000004, 2020000006, 2020000007, 2020000008, 2020000009, 2020000010, 2020000011, 2020000012, 2020000013, 2020000014, 2020000015, 2020000016, 2020000017, 2020000018, 2020000019, 2020000022, 2020000023, 2020000024, 2020000025, 2020000026, 2020000028, 2020000029, 2020000034, 2020000035, 2020000036, 2020000038, 2020000039, 2020000041, 2020000042, 2020000045, 2020000046, 2020000049, 2020000052, 2020000053, 2020000055, 2020000056, 2020000061, 2020000066, 2020000067, 2020000072, 2020000076, 2020000091, 2020000093, 2020000097, 2020000102, 2020000105, 2020000106, 2020000109, 2020000112, 2020000115, 2020000117, 2020000139, 2020000146, 2020000160, 2020000162, 2020000165, 2020000170, 2020000171, 2020000175, 2020000178, 2020000202, 2020000209, 2020000223, 2020000234, 2020000246, 2020000258, 2020000260, 2020000267, 2020000273, 2020000276, 2020000296, 2020000303, 2020000305, 2020000309, 2020000310, 2020000317, 2020000318, 2020000319, 2020000324, 2020000325, 2020000328, 2020000332, 2020000340, 2020000352, 2020000353, 2020000354, 2020000358, 2020000365, 2020000366, 2020000372, 2020000373, 2020000379, 2020000380, 2020000383, 2020000385, 2020000386, 2020000393]}
```
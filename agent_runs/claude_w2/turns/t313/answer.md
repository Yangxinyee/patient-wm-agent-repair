The fwd_9 stays are exactly sel1, so the comparison is paired and conclusive.

| Candidate | Value NLL | Change vs original (1.52793) |
|---|---|---|
| cand3 (numerics ×5 + head readout) | 1.57204 | +0.0441 |
| cand2 (bins ×3) | 1.53339 | +0.0055 |
| numerics ×3 alone | 1.52582 | −0.0021 |

cand3 fails the 0.005 gate by about 9×. The damage is entirely in glucose: finger 1.559 → 1.659 and lab 1.227 → 1.324. MAP is slightly better (1.3894 → 1.3886, 1.7687 → 1.7672). So the dose numerics ×5 look harmless or helpful, and the gated insulin readout on head unit 109 is the culprit. It is very large: bias 24, an input row about 2× the norm of the whole layer, and a logit tilt over all 17 bins. A likely cause is that its large input weights amplify the 4 prior-z draws the head makes, which my local design probably did not include.

Before spending one of my last 6 Python runs, I need to know how much rollout effect this readout buys at full strength. I also need the NE effect of numerics ×5 alone, which the glucose-gated readout should not touch. This dose_edit pass on the same 100 stays costs no Python run.

- **If insulin ×2 is far beyond the original's −2.14 (say ≤ −6):** a scaled-down readout (output and compensation × s ≈ 0.25, cost roughly s² × 0.044) is worth keeping.
- **If it is only about −2 to −3:** I will drop the head edits and keep only the numeric amplification, with k chosen to put NE ×2 / ×½ near my record-based estimates of +4.7 / −2.4.

```forward
{"type": "dose_edit", "model": "edited", "split": "tuning", "stays": [2020000000, 2020000002, 2020000003, 2020000004, 2020000006, 2020000007, 2020000008, 2020000009, 2020000010, 2020000011, 2020000012, 2020000013, 2020000014, 2020000015, 2020000016, 2020000017, 2020000018, 2020000019, 2020000022, 2020000023, 2020000024, 2020000025, 2020000026, 2020000028, 2020000029, 2020000034, 2020000035, 2020000036, 2020000038, 2020000039, 2020000041, 2020000042, 2020000045, 2020000046, 2020000049, 2020000052, 2020000053, 2020000055, 2020000056, 2020000061, 2020000066, 2020000067, 2020000072, 2020000076, 2020000091, 2020000093, 2020000097, 2020000102, 2020000105, 2020000106, 2020000109, 2020000112, 2020000115, 2020000117, 2020000139, 2020000146, 2020000160, 2020000162, 2020000165, 2020000170, 2020000171, 2020000175, 2020000178, 2020000202, 2020000209, 2020000223, 2020000234, 2020000246, 2020000258, 2020000260, 2020000267, 2020000273, 2020000276, 2020000296, 2020000303, 2020000305, 2020000309, 2020000310, 2020000317, 2020000318, 2020000319, 2020000324, 2020000325, 2020000328, 2020000332, 2020000340, 2020000352, 2020000353, 2020000354, 2020000358, 2020000365, 2020000366, 2020000372, 2020000373, 2020000379, 2020000380, 2020000383, 2020000385, 2020000386, 2020000393]}
```
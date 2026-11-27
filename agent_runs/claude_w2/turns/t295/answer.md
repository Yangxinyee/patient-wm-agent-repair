The joint fit is informative:
- **Factual weight μ 0.1–0.3, gain 1:** the readout improves factual glucose prediction on held-out stays (mean-squared error about −13 to −20 (mg/dL)²). The model under-uses information it already holds.
- **Larger gains:** they cost fit. At μ 0.3, gain 2 adds +29.
- **Counterfactual response:** only about 0.5 of the missing one-step effect at gain 1.
- **Factual-fit limit:** budget allows a gain of about 1.5 (MSE change about −4). The insulin response will improve but likely stay below the truth.

I have 19 runs and 46 forward passes, so I'll build the edit now and calibrate with the forward tool. The edit:
1. Repurpose the least important hidden unit of the value head's MLP as a gated, near-linear unit.
   - Its input weights are G·(gate direction) + w/κ. The gate is fitted so it is +1 on glucose codes and −1 on MAP codes, with G = 20, putting glucose rows in GELU's linear range and MAP rows at zero.
   - Its output is a logit tilt along the glucose bin centres, scaled so one unit shifts the mean by about 1 mg/dL.
2. Move its old contribution into the output bias.
3. Subtract the constant G·tilt from the glucose rows of the transition table.

I'll verify on `D4` rows:
- the shift per arm matches λ·s;
- MAP logits are unchanged;
- factual bin NLL does not get worse.

Then I'll save the edit.

```python
nm_vb = [n for n in NAMES if 'value_bin' in n]; print({n: SHAPES[n] for n in nm_vb})
```
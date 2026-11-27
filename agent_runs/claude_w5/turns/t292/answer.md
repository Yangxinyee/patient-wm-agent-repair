The documentation confirms the tool doubles or halves both rate and dose, and that is what my decoder was fitted on. Locally the "both" variant is linear and symmetric: per unit of rate, dx is +0.905 for x2 and −0.472 for x1/2, against +1 and −0.5 expected.

So the tool's hour-13 asymmetry (x2 −3.08, x1/2 +0.09) does not come from my edit's logic or the dose-edit semantics as I simulate them. The tool's computation of the edited model must differ from mine. The fwd_5 comparison showed exactly that kind of gap, but it was made without the kv edit.

Next I'll re-run the states comparison on the same 13 tuning stays with the full edit now saved, and compare the tool's teacher-forced states with my local `WE` states.

- If they match to the original model's level (about 1e-4), the tool core equals mine, and the asymmetry must come from the head or the rollout.
- If the edited states still differ, the large kv weights make the tool's float32 GPU computation diverge from my float64 one, and I will need a numerically gentler stage 1.

```forward
{"type": "states", "model": "both", "split": "tuning", "stays": [2020000112, 2020000033, 2020000034, 2020000066, 2020000036, 2020000117, 2020000038, 2020000054, 2020000070, 2020000090, 2020000092, 2020000045, 2020000014]}
```
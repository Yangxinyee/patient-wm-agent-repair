Dose innovations are weakly recoverable from the raw head input, but nearly lost in the existing hidden features. That explains the failed final-layer probe. I’ll prepare a glucose-only decoder from the raw input and make it insensitive to the uncertain register layout; retained decoding accuracy and low reconstruction error would support a reversible test.

```python
import numpy as np
from scipy.linalg import eigh
from sklearn.svm import LinearSVC
from sklearn.metrics import r2_score
from pathlib import Path

# New decoder directions cannot depend on the disputed numeric-register layout.
register_basis = W["last_numeric.weight"].numpy().astype(np.float64)
Ureg, sreg, _ = np.linalg.svd(register_basis, full_matrices=False)
Ureg = Ureg[:, sreg > sreg.max() * 1e-7]
register_null = np.eye(192) - Ureg @ Ureg.T

raw_input_sets = {s: fs["head_input"] for s, fs in decode_feature_sets.items()}
protected_input_sets = {s: xx @ register_null for s, xx in raw_input_sets.items()}
glucose_masks = {
    s: m.event.isin(["glucose_finger", "glucose_lab"]).to_numpy()
    for s, m in decode_meta_sets.items()
}
valid_glucose_masks = {
    s: glucose_masks[s] & np.asarray(decode_dose_sets[s]["valid"], bool)
    for s in decode_meta_sets
}

new_decoder_candidates = {}
for protected in [False, True]:
    fs = protected_input_sets if protected else raw_input_sets
    A = fs["first"][valid_glucose_masks["first"]]
    B = fs["extra"][valid_glucose_masks["extra"]]
    C = fs["tuning"][valid_glucose_masks["tuning"]]
    yy = unit_first[valid_glucose_masks["first"]]
    yb = unit_extra[valid_glucose_masks["extra"]]
    yc = unit_tuning[valid_glucose_masks["tuning"]]
    loc0, scale0 = A.mean(0), np.maximum(A.std(0), 0.05)
    A = (A - loc0) / scale0
    B, C = (B - loc0) / scale0, (C - loc0) / scale0
    ev, U0 = eigh(A.T @ A)
    ev = np.maximum(ev, 0)
    cross = U0.T @ (A.T @ (yy - yy.mean()))
    trials = []
    for alpha in [0.01, 0.1, 1, 10, 100, 1000]:
        ww0 = U0 @ (cross / (ev + alpha))
        pb, pc = yy.mean() + B @ ww0, yy.mean() + C @ ww0
        trials.append((r2_score(yb, pb), alpha, r2_score(yc, pc), ww0))
    best = max(trials, key=lambda t: t[0])
    ww0 = best[3] / scale0
    if protected:
        ww0 = register_null @ ww0
    bb0 = float(yy.mean() - loc0 @ (best[3] / scale0))
    new_decoder_candidates[protected] = (ww0, bb0, best[1])
    print("protected", protected, "alpha", best[1],
          "extra R2", best[0], "tuning R2", best[2],
          "tuning output SD", np.std(raw_input_sets["tuning"] @ ww0 + bb0),
          "register sensitivity", np.linalg.norm(register_basis.T @ ww0))

gate_X = np.concatenate([protected_input_sets["first"],
                         protected_input_sets["extra"]])
gate_y = np.concatenate([glucose_masks["first"], glucose_masks["extra"]])
gate_fit = LinearSVC(C=1.0, dual=False, max_iter=10000)
gate_fit.fit(gate_X, gate_y.astype(int))
gate_w = register_null @ gate_fit.coef_[0]
gate_b = float(gate_fit.intercept_[0])
for s, xx in raw_input_sets.items():
    margin = xx @ gate_w + gate_b
    correct = (margin > 0) == glucose_masks[s]
    print("gate", s, "accuracy", correct.mean(),
          "signed margin quantiles",
          np.quantile(margin * (2 * glucose_masks[s].astype(int) - 1),
                      [0, .01, .1, .5]))

original_head_hidden = np.concatenate([
    decode_feature_sets["first"]["head_hidden"],
    decode_feature_sets["extra"]["head_hidden"]
])
original_output_weight = load_original("attr_heads.value_bin.2.weight")
if isinstance(original_output_weight, torch.Tensor):
    original_output_weight = original_output_weight.numpy()
neuron_influence = (original_head_hidden.std(0)
                    * np.linalg.norm(original_output_weight, axis=0))
print("LOW-INFLUENCE NEURONS",
      [(int(i), float(neuron_influence[i]),
        float(np.linalg.norm(original_output_weight[:, i])))
       for i in np.argsort(neuron_influence)[:12]])
print("RANK TEMPLATE", np.asarray(rank_template))
print("HEAD SHAPES", [(n, SHAPES[n]) for n in NAMES
                     if n.startswith("attr_heads.value_bin")])

p = Path("/work/notes.md")
p.write_text(p.read_text() + """
## Raw-input decoder recovery; Python 64, forward 7/50
- Insulin dose innovations: head input held-out R2 ~.05 extra / .094 tuning.
- Existing hidden-feature innovation decoder ~zero R2; earlier probe had no leverage.
- Direct insulin dose decoding R2 ~.82; action information is not entirely absent.
- Preparing a gated raw-input decoder; project away numeric-register directions.
- No current edits. GPU factual and dose rollouts will determine acceptance.
""")
```

The state penalty cuts held-out perturbations to about 0.23 while retaining most of the estimated doubling response and improving the halving response. I’ll check the nonlinear anchored decoder, then install this candidate for GPU validation. Small anchored changes would support the revised approximation, but only the GPU factual test can establish feasibility.

```python
import numpy as np
import torch
from scipy.special import logsumexp

gamma151 = 0.1
coef151 = fits150[gamma151]
ne_delta151 = Bdir141 @ coef151 @ Binput142.T
ne_shift151 = (Fslots149 @ coef151.T) @ Pdir143.T
baseline_shift151 = (
    (EF141 @ baseline_delta147.astype(np.float64).T)
    * delta148[None, None, :]
) @ cout148.T

parsed_base151 = extract_readout143(base_result143)
parts_base151 = list(parsed_base151)
logit_index151 = next(
    i for i, x in enumerate(parts_base151)
    if np.asarray(x).shape == base_logits143.shape
    and np.max(np.abs(np.asarray(x) - base_logits143)) < 1e-5
)
mean_index151 = next(
    i for i, x in enumerate(parts_base151)
    if np.asarray(x).shape == base_mean143.shape
    and np.max(np.abs(np.asarray(x) - base_mean143)) < 1e-5
)

hbase151, obase151 = coda_input134(
    anchor_x143 + torch.from_numpy(baseline_shift151.astype(np.float32))
)
hnew151, onew151 = coda_input134(
    anchor_x143
    + torch.from_numpy((baseline_shift151 + ne_shift151).astype(np.float32))
)
parsed_baseline151 = extract_readout143(readout125(hbase151, obase151))
parsed_new151 = extract_readout143(readout125(hnew151, onew151))
lb151 = np.asarray(parsed_baseline151[logit_index151], dtype=np.float64)
ln151 = np.asarray(parsed_new151[logit_index151], dtype=np.float64)
mb151 = np.asarray(parsed_baseline151[mean_index151], dtype=np.float64)
mn151 = np.asarray(parsed_new151[mean_index151], dtype=np.float64)
logpb151 = lb151 - logsumexp(lb151, axis=1, keepdims=True)
logpn151 = ln151 - logsumexp(ln151, axis=1, keepdims=True)
kl151 = np.sum(np.exp(logpb151) * (logpb151 - logpn151), axis=1)

print("NONLINEAR ANCHORED NE-INCREMENT CHECK", {
    "heldout_KL": float(np.mean(kl151[held144])),
    "heldout_mean_shift_rms":
        float(np.sqrt(np.mean((mn151[held144] - mb151[held144]) ** 2))),
    "NE_weight_norm": float(np.linalg.norm(ne_delta151)),
    "pre_coda_state_rms": float(np.sqrt(np.mean(ne_shift151 ** 2))),
})
assert np.isfinite(ln151).all() and np.isfinite(ne_delta151).all()

total_delta151 = baseline_delta147.astype(np.float64) + ne_delta151
candidate151 = (original_b147.astype(np.float64) + total_delta151).astype(np.float32)
save_edit("core.b_in.weight", candidate151)
np.savez(
    "/work/ne_candidate151.npz",
    ne_delta=ne_delta151.astype(np.float32),
    total_delta=total_delta151.astype(np.float32),
    gamma=np.array(gamma151),
)
print("INSTALLED baseline77 + state-penalized NE correction, gamma=.1")
print("Saved-weight maximum error",
      float(np.max(np.abs(np.asarray(load("core.b_in.weight")) - candidate151))))

with open("/work/notes.md", "a") as f151:
    f151.write(
        "\n## Run 151: state-penalized NE candidate installed\n"
        "Installed original b_in + baseline77 + "
        "Bdir141 @ fits150[.1] @ Binput142.T. "
        f"Anchored held-out incremental KL={np.mean(kl151[held144]):.7f}; "
        "this is a local diagnostic, not full factual NLL. "
        "GPU factual and dose validation pending. "
        "Python runs: 151; forwards: 16.\n"
    )
```
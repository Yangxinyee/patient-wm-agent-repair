The current weights are exactly the insulin baseline plus 0.35 of the old norepinephrine trial. The new sensitivity directions include recurrent scaling, so they must not be used directly as weight directions. I’ll verify that mapping and, if it matches, install the intermediate-penalty candidate. GPU tests will determine whether its local safeguards actually hold.

```python
import numpy as np

delta148 = np.logaddexp(
    0.0, np.asarray(load_original("core.delta_in"), dtype=np.float64)
).reshape(192)
cout148 = np.asarray(load_original("core.c_out.weight"), dtype=np.float64)

mapped_directions148 = cout148 @ (delta148[:, None] * Bdir141)
fresh_features148 = EF141.mean(axis=1) @ Binput142
ne_delta148 = Bdir141 @ fits144[10.0] @ Binput142.T

checks148 = {
    "sensitivity_map_max_error":
        float(np.max(np.abs(mapped_directions148 - Pdir143))),
    "bundle_feature_max_error":
        float(np.max(np.abs(fresh_features148 - Fmean142))),
    "reading_feature_max_error":
        float(np.max(np.abs(fresh_features148[src141] - Fread144))),
    "admission_null_max_error":
        float(np.max(np.abs(admission142 @ Binput142))),
    "weight_direction_gram_error":
        float(np.linalg.norm(Bdir141.T @ Bdir141 - np.eye(14))),
    "candidate_norm": float(np.linalg.norm(ne_delta148)),
    "reported_norm": float(metrics144[10.0]["norm"]),
}
print("MAPPING CHECKS", checks148)

ready148 = (
    checks148["sensitivity_map_max_error"] < 1e-4
    and checks148["bundle_feature_max_error"] < 1e-4
    and checks148["reading_feature_max_error"] < 1e-4
    and checks148["admission_null_max_error"] < 1e-4
    and checks148["weight_direction_gram_error"] < 1e-4
    and abs(checks148["candidate_norm"] - checks148["reported_norm"]) < .01
)

if ready148:
    total_delta148 = baseline_delta147.astype(np.float64) + ne_delta148
    candidate148 = (original_b147.astype(np.float64) + total_delta148).astype(np.float32)
    save_edit("core.b_in.weight", candidate148)
    np.savez(
        "/work/ne_candidate148.npz",
        ne_delta=ne_delta148.astype(np.float32),
        total_delta=total_delta148.astype(np.float32),
        anchor=np.array(10.0),
    )
    print("INSTALLED: original + baseline77 + multi-direction NE candidate, anchor=10")
    print("Saved-weight maximum error",
          float(np.max(np.abs(np.asarray(load("core.b_in.weight")) - candidate148))))
else:
    print("NOT INSTALLED: mapping verification failed; previous weights retained.")

with open("/work/notes.md", "a") as f148:
    f148.write(
        "\n## Run 148: verified multi-direction NE trial\n"
        f"Mapping checks: {checks148}. Installed={ready148}. "
        "If installed, composition is original b_in + baseline77 + "
        "Bdir141 @ fits144[10] @ Binput142.T, replacing—not adding to—the old NE trial. "
        "Candidate remains unaccepted until GPU factual and paired-rollout tests. "
        "Python runs: 148; forwards: 15.\n"
    )
```
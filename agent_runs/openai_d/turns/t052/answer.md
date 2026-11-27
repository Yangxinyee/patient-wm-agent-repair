The states align with the preceding bundles, and the reading query retains insulin information that becomes harder to decode after adding the last-reading register. I’ll test a decoder that ignores the register’s input directions. If it generalizes across stays, it could support a small head correction; otherwise I’ll move upstream to the action pathway.

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

print("=== LATENT AND QUERY TENSORS ===")
for rec_name in NAMES:
    if any(s in rec_name.lower() for s in ["prior", "post", "z_in", "z_head", "query_norm"]):
        print(rec_name, SHAPES[rec_name])

rec_ln = np.asarray(load_original("last_numeric.weight"))
rec_vb = np.asarray(load_original("embed.bins.value.weight"))
rec_basis = np.concatenate([rec_ln, rec_vb.T], axis=1)
rec_u, rec_s, _ = np.linalg.svd(rec_basis, full_matrices=False)
rec_rank = int(np.sum(rec_s > rec_s[0] * 1e-6))
rec_reg_projector = np.eye(192) - rec_u[:, :rec_rank] @ rec_u[:, :rec_rank].T
print("\nRegister subspace rank:", rec_rank)

rec_decode_tests = {}
for rec_drug, rec_event in [
    ("insulin", "glucose_finger"), ("norepinephrine", "map_arterial")
]:
    rec_a = repair_aligned[rec_drug]
    rec_mask = ((rec_a["event"].to_numpy() == rec_event)
                & np.isfinite(rec_a["rd"].to_numpy()))
    rec_indices = np.flatnonzero(rec_mask)
    rec_y = rec_a["rd"].to_numpy()[rec_indices]
    rec_groups = rec_a["stay_id"].to_numpy()[rec_indices]
    rec_features = {
        "query": np.asarray(_r40_q)[rec_indices],
        "query_no_register_axes": np.asarray(_r40_q)[rec_indices] @ rec_reg_projector,
        "with_code": np.asarray(repair_wc_mean)[rec_indices],
        "with_code_no_register_axes": np.asarray(repair_wc_mean)[rec_indices] @ rec_reg_projector,
    }
    print("\n", rec_drug, "active primary rows:", len(rec_y),
          "innovation SD:", float(np.std(rec_y)))
    rec_decode_tests[rec_drug] = {}
    for rec_label, rec_x in rec_features.items():
        rec_scores = []
        for rec_alpha in [.001, .01, .1, 1., 10., 100.]:
            rec_pred = np.zeros(len(rec_y))
            for rec_tr, rec_va in GroupKFold(4).split(rec_x, rec_y, rec_groups):
                rec_fit = Ridge(alpha=rec_alpha).fit(rec_x[rec_tr], rec_y[rec_tr])
                rec_pred[rec_va] = rec_fit.predict(rec_x[rec_va])
            rec_scores.append((rec_alpha, float(r2_score(rec_y, rec_pred)),
                               float(np.corrcoef(rec_y, rec_pred)[0, 1])))
        rec_best = max(rec_scores, key=lambda q: q[1])
        rec_decode_tests[rec_drug][rec_label] = rec_best
        print(rec_label, "best (alpha, R2, corr):", rec_best)

with open("/work/notes.md", "a") as rec_f:
    rec_f.write("\n## Run 46: register-invariant decoding\n"
                + repr(rec_decode_tests)
                + "\nNo edits. Tested whether last-reading input directions "
                  "explain the loss of innovation decodability in the head.\n")
```
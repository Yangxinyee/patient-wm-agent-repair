The discrepancy is resolved: the older targets averaged stays equally, while the later ones pooled stay-hours. Under either convention, insulin remains under-responsive. I will recheck and restore the stronger-insulin candidate, which previously had a safer factual-loss margin, then compare all 400 rollouts. Better insulin alignment would support keeping it; worsening would favor reverting. Norepinephrine may weaken slightly.

```python
import numpy as np
import pandas as pd
import torch

audit299_parts = []
for folder in ("fwd_27", "fwd_29", "fwd_30", "fwd_31"):
    original = pd.read_csv("/work/" + folder + "/factual_original.csv")
    candidate = pd.read_csv("/work/" + folder + "/factual_edited.csv")
    assert len(original) == len(candidate)
    keys = [
        c for c in ("stay_id", "hour", "event", "observed", "value")
        if c in original.columns and c in candidate.columns
    ]
    assert "stay_id" in keys
    assert original[keys].equals(candidate[keys]), folder

    choices = [
        c for c in ("nll", "value_nll", "fine_bin_nll")
        if c in original.columns and c in candidate.columns
    ]
    if not choices:
        choices = [
            c for c in original.columns
            if "nll" in c.lower() and c in candidate.columns
        ]
        assert len(choices) == 1, list(original.columns)
    nll_column = choices[0]
    part = pd.DataFrame({
        "stay_id": original["stay_id"].to_numpy(),
        "original_nll": original[nll_column].to_numpy(),
        "edited_nll": candidate[nll_column].to_numpy(),
    })
    part["delta_nll"] = part["edited_nll"] - part["original_nll"]
    audit299_parts.append(part)

audit299_paired = pd.concat(audit299_parts, ignore_index=True)
assert audit299_paired["stay_id"].nunique() == 400
assert len(audit299_paired) == 18654

audit299_stay_stats = audit299_paired.groupby("stay_id")[
    "delta_nll"
].agg(["sum", "count"])
audit299_rng = np.random.default_rng(299)
audit299_draws = audit299_rng.integers(
    0, len(audit299_stay_stats), size=(10000, len(audit299_stay_stats))
)
audit299_boot = (
    audit299_stay_stats["sum"].to_numpy()[audit299_draws].sum(axis=1)
    / audit299_stay_stats["count"].to_numpy()[audit299_draws].sum(axis=1)
)
audit299_delta = float(audit299_paired["delta_nll"].mean())
audit299_quantiles = np.quantile(audit299_boot, [.025, .5, .95, .975])

print("=== STRONGER-INSULIN CANDIDATE: EXACT PRIOR FACTUAL VALIDATION ===")
print(audit299_paired[
    ["original_nll", "edited_nll", "delta_nll"]
].mean().to_string())
print("Readings:", len(audit299_paired))
print("Stays:", audit299_paired["stay_id"].nunique())
print("Bootstrap [.025, .5, .95, .975]:", audit299_quantiles.tolist())

audit299_margin_weight = audit298_archives[
    "margin_validated400.npz"
]["core_b_in"].copy()
audit299_stronger_weight = audit298_archives[
    "validated_current400.npz"
]["core_b_in"].copy()
audit299_glucose = audit298_archives[
    "validated_current400.npz"
]["glucose_correction"]
audit299_map = audit298_archives[
    "validated_current400.npz"
]["map_correction"]

print("\n=== ARCHIVED EDIT FORMULA CHECK ===")
print("Stronger-insulin formula maximum discrepancy:", float(np.max(np.abs(
    audit299_stronger_weight
    - (audit298_original + 1.15 * audit299_glucose + audit299_map)
))))
print("Margin formula maximum discrepancy:", float(np.max(np.abs(
    audit299_margin_weight
    - (audit298_original + audit299_glucose + 1.4 * audit299_map)
))))

audit299_switched = (
    audit299_delta < 0.005
    and float(audit299_quantiles[-1]) < 0.005
)
np.savez(
    "/work/final_switch_audit.npz",
    margin_core_b_in=audit299_margin_weight,
    stronger_insulin_core_b_in=audit299_stronger_weight,
    stronger_insulin_value_nll_delta=np.array(audit299_delta),
    stronger_insulin_bootstrap_quantiles=audit299_quantiles,
    margin_value_nll_delta=np.array(0.0033847968264179265),
)

if audit299_switched:
    save_edit(
        "core.b_in.weight",
        torch.as_tensor(audit299_stronger_weight, dtype=torch.float32).clone(),
    )
    assert np.array_equal(
        audit298_numpy(load("core.b_in.weight")),
        audit299_stronger_weight,
    )
    print("\nRestored exact, previously validated stronger-insulin candidate.")
else:
    print("\nSafety check failed; retained the validated margin candidate.")

print("Currently edited tensors:", edited())
print("New forward baseline folder: fwd_45; subsequent successful calls start at fwd_46.")

with open("/work/notes.md", "a") as file:
    file.write(
        "\n## Run 299: resolve averaging discrepancy and compare eligible candidates\n"
        "- Run-132 targets used equal-stay averaging; later summaries pooled stay-hours. "
        "Same-row convolution discrepancies are at numerical precision only.\n"
        "- Full-400 six-lag targets: hour-weighted insulin x2/half "
        "-12.720408/+6.360204 and NE x2/half +4.169603/-2.084802; "
        "stay-weighted -10.933000/+5.466500 and +4.090764/-2.045382.\n"
        "- Independently reconstructed stronger-insulin candidate factual validation "
        "from fwd_27/29/30/31: "
        f"delta {audit299_delta:.12f} over {len(audit299_paired)} readings; "
        f"bootstrap quantiles {audit299_quantiles.tolist()}.\n"
        f"- Exact stronger-insulin archive restored: {audit299_switched}. "
        "Only core.b_in.weight is edited; no gradients or optimizer used.\n"
        "- Final selection will compare identical 400-stay rollout cohorts, "
        "prioritizing the stated main insulin readout while documenting the MAP tradeoff.\n"
        "- Both candidate weights preserved in final_switch_audit.npz.\n"
        "- Python runs: 299; forward passes: 44.\n"
    )
```
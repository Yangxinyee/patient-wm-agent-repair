The panel confirms that doses follow the current readings, so raw dose–outcome associations are not reliable effect estimates. Insulin threshold estimates are consistently negative, but the output was truncated before the dynamic and norepinephrine diagnostics. I’ll retrieve those compactly and check units; agreement between threshold and residual-dose estimates would strengthen the causal target.

```python
import numpy as np
from pathlib import Path

print("=== COMPACT CAUSAL RESULTS ===")
print(rd_table.tail(6).round(5).to_string(index=False))
print("Dynamic fitted parameters:", dyn_fit.x)
for key, value in dynamic_rd.items():
    print("dynamic_rd", key, str(value)[:350])
print("NE feature names:", feature_names)
print("NE_DML:", NE_DML, "dose residual SD:", np.std(rd),
      "outcome residual SD:", np.std(ry))
for drug_name in ["insulin", "norepinephrine"]:
    doses = panel.loc[panel[drug_name] > 0, drug_name]
    print(drug_name, "positive count", len(doses),
          "quantiles", doses.quantile([0, .1, .25, .5, .75, .9, .99, 1]).to_dict(),
          "top rates", doses.value_counts().head(12).to_dict())
print("dne vs raw:", dne[:8], sne["norepinephrine"].to_numpy()[:8])
print("yne vs raw:", yne[:8], sne["map_arterial_next"].to_numpy()[:8])
print("Rule rows:", str(rule_rows)[:1500])

print("\n=== ACTION, POSTERIOR, AND HEAD INPUT TENSORS ===")
prefixes = ("encode_action.", "z_post.", "z_prior.", "z_in.", "z_head.",
            "last_", "query_", "gap.", "embed.numeric", "type_emb")
for name in NAMES:
    if name.startswith(prefixes):
        print(name, SHAPES[name])
print("\n=== VOCAB DETAILS ===")
print("keys:", list(VOCAB))
print("config:", VOCAB.get("config"))
print("codes:", VOCAB.get("codes"))
for key in VOCAB:
    if key not in {"config", "codes"}:
        value = VOCAB[key]
        print(key, str(value)[:1800])

print("\n=== FORWARD FILE HEADERS ===")
print(list(Path("/work/fwd_2").iterdir()))
factual0 = pd.read_csv("/work/fwd_2/factual_original.csv")
print(factual0.columns.tolist())
print(factual0.head(3).to_string(index=False))
print("Validation IDs:", VALID_IDS[:10])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 9: compact units and causal audit\n"
            "Retrieved dynamic identification, dose units/support, exact action/posterior "
            "shapes, and head-input configuration. No weights edited yet.\n"
            "Python runs: 9. Forward requests: 2.\n")
```
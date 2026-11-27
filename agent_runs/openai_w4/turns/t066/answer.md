The latest tests show that measurement posteriors can be reconstructed accurately once the learned empty-action encoding is included. That resolves a concrete encoding error, but not the repair itself. I’ll use the corrected recipes to check bundle alignment and the retained dose residuals; matching recorded doses and exported states would justify the next controlled experiment.

```python
import json
import inspect

print("=== MODEL CONFIG ===")
print(VOCAB["config"])
print("=== ENCODER TENSORS ===")
for n57 in NAMES:
    if n57.startswith(("encode.", "encode_action.")):
        print(n57, SHAPES[n57])
print("=== FUNCTION CONSTANTS ===")
for n57 in ("atom_embedding_repair", "rp_action_encoder",
            "pre_context_repair", "build_head_dataset_fixed"):
    fn57 = globals()[n57]
    print(n57, repr(fn57.__code__.co_consts)[:2600])
print("=== CORRECT MEASUREMENT RECIPES ===")
for event57, recipe57 in r52_best.items():
    measurement_recipe_repair[event57] = (recipe57[1], recipe57[2])
    repair_units[event57] = recipe57[1]
print(measurement_recipe_repair)

print("=== BASELINE SUMMARY ===")
print(json.loads(Path("/work/fwd_1/summary.json").read_text()))
print("=== RESIDUAL ALIGNMENT ===")
print("HX index equals pan:", HX.index.equals(pan.index))
for drug57 in ("insulin", "norepinephrine"):
    rr57 = dml_residuals[drug57]
    ii57 = rr57["indices"][:8]
    print(drug57, "indices", ii57)
    print(pan.iloc[ii57].to_string())
    print("stored dose", rr57["dose"][:8])
    print("innovation", rr57["rd"][:8])
    print("models", [(type(m).__name__, getattr(m, "n_features_in_", None))
                     for m in dml_models[drug57]])
print("=== STATE/RECORD ALIGNMENT SAMPLE ===")
print({k: (v.shape if hasattr(v, "shape") else type(v).__name__)
       for k, v in repair_bank.items()})
print(repair_records.head(8).to_string(index=False))
print("state rows:",
      list(zip(repair_bank["stay_id"][:8], repair_bank["hour"][:8],
               repair_bank["n_atoms"][:8])))
with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 57\nCorrected measurement units/bin sides from run-52 best recipes (posterior RMSE about 2.5e-5); learned size-bias empty-action encoding is essential. Checking exact encoder and causal-panel alignment. No weights edited.\n")
```
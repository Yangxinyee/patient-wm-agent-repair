The temporary embedding edit is now removed. The architecture shows that drug inputs affect both the posterior and recurrent state, so changing the value head alone may miss the treatment pathway. I’ll recover the effect estimates and baseline rollouts next; consistent estimates across channels and specifications would support a causal repair.

```python
import pandas as pd, json

print("=== RECORD EFFECT ESTIMATES ===")
print(pd.DataFrame(causal_estimates).to_string(index=False))

print("\n=== POSTERIOR DECODER RESULTS ===")
print(pd.DataFrame(repair_post_results).to_string(index=False))
print("Effect diagnostics:", repair_effect_diagnostics)

print("\n=== RELEVANT WEIGHT SHAPES ===")
for name in NAMES:
    if any(s in name for s in
           ("z_post", "z_prior", "z_in", "z_head", "attr_heads.value_bin",
            "last_numeric", "embed.numeric", "encode_action")):
        print(name, SHAPES[name])

print("\n=== BASELINE ROLLOUT FILES ===")
for folder in ("/work/fwd_1", "/work/fwd_3", "/work/fwd_5", "/work/fwd_6"):
    if os.path.isdir(folder):
        for fn in os.listdir(folder):
            if "dose_edit" in fn and fn.endswith(".csv"):
                tab = pd.read_csv(os.path.join(folder, fn))
                print(folder, fn, tab.shape, list(tab.columns))
                print(tab.head(5).to_string(index=False))

print("\n=== EARLY NOTEBOOK EVIDENCE ===")
notes_recovery = open("/work/notes.md").read()
print(notes_recovery[:7500])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 157\n"
            "Original weights restored. Recovering record-derived causal "
            "estimates, posterior decoder validation, baseline rollouts, "
            "and exact editable pathways before choosing a candidate. "
            "Counts: 157 Python runs, 8 forward passes.\n")
```
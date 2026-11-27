The architecture shows that the current edit changes a recurrent state-update matrix; it does not alter the factual prediction heads directly. The saved norepinephrine trials add substantial perturbations, so magnitude alone is not a good guide. I’ll recover their measured trade-offs and the original causal evidence. That will show whether to refine the state edit or seek factual-fit headroom elsewhere.

```python
import re, pandas as pd

notes216 = open("/work/notes.md").read()
print("=== INITIAL IDENTIFICATION EVIDENCE ===")
print(notes216[:3300])

print("\n=== SELECTED EXPERIMENT NOTES ===")
budget216 = 4400
for section216 in re.split(r"(?=^## )", notes216, flags=re.M):
    heading216 = section216.split("\n", 1)[0]
    if re.search(r"\b(76|77|78|155|156|158|175|183|195|213)\b", heading216):
        excerpt216 = section216[:min(850, budget216)]
        print(excerpt216)
        budget216 -= len(excerpt216)
        if budget216 <= 0:
            break

print("\n=== VALUE-HEAD TENSORS ===")
for name216 in NAMES:
    if "attr_heads.value_bin" in name216 or "embed_transition.value_bin" in name216:
        print(name216, SHAPES[name216])

print("\n=== LATE DECODER HELPERS ===")
for name216 in ["reading_metadata162", "build_static167", "decode_states175"]:
    f216 = globals().get(name216)
    if callable(f216):
        print(name216, "referenced names:", f216.__code__.co_names)
        print("string constants:",
              [x for x in f216.__code__.co_consts if isinstance(x, str)][:35])

print("\n=== FACTUAL OUTPUT SCHEMA ===")
p216 = "/work/fwd_10/factual_edited.csv"
if os.path.exists(p216):
    df216 = pd.read_csv(p216)
    print(df216.shape, list(df216.columns))
    print(df216.head(2).to_dict("records"))

with open("/work/notes.md", "a") as f216:
    f216.write(
        "\n## Run 216: recover evidence and headroom options\n"
        "Architecture confirms b_in is the recurrent state-update injection point "
        "and factual value predictions use a separate 17-class MLP plus register "
        "transition logits. Recovering baseline/NE validation and causal "
        "identification evidence, with bounded output. No weights changed. "
        "Python runs: 216; forwards: 23.\n"
    )
```

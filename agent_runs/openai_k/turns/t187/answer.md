The high-penalty decoder’s noise corresponds to corrections of roughly half a glucose unit and a few tenths of a MAP unit after rescaling—potentially small enough to test. I’ll recover the value-head inputs and dose alignment, then fit a correction that penalizes factual changes. I’ll reject it if held-out accuracy requires excessive factual noise.

```python
import dis, re, numpy as np

print("=== ACTION / BUNDLE ALIGNMENT ===")
for k in ["causal_rows", "rr33_u_all", "r49_register", "r49_hours",
          "A164", "Y164", "dc151", "dc156", "lookup157",
          "o164", "h164", "out_original154", "h_original154"]:
    if k in globals():
        v = globals()[k]
        print(k, r170_brief(v))
        if hasattr(v, "columns"):
            print(" columns", list(v.columns))
            print(v.head(2).to_string(index=False))

print("\n=== RETAINED REGISTER CALIBRATION ===")
for k, v in list(globals().items()):
    if re.search(r"(118|119|120)$", k) and not callable(v):
        if any(s in k.lower() for s in [
            "layout", "param", "sol", "fit", "opt", "proj", "feature",
            "res", "loss", "err", "best", "query", "numeric"
        ]):
            print(k, r170_brief(v))

print("\n=== READOUT INPUT CONSTRUCTION ===")
r171_instructions = list(dis.get_instructions(r49_readouts))
r171_text = []
for ins in r171_instructions:
    if ins.starts_line is not None:
        r171_text.append(f"\nL{ins.starts_line}:")
    if ins.opname not in ["RESUME", "PRECALL", "PUSH_NULL", "COPY", "SWAP"]:
        r171_text.append(f"{ins.opname} {ins.argrepr}")
print(" | ".join(r171_text)[-4700:])

print("\n=== SIMPLE HEAD AND REGISTRY SHAPES ===")
for k in NAMES:
    if any(s in k for s in ["last_numeric", "last_status", "embed.bins.value",
                            "embed.bins.rate", "embed_transition.value_bin"]):
        print(k, SHAPES[k])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 171: prepare factual-penalized head-feature assay\n"
        "- High-penalty dose decoding, after slope rescaling, implies roughly "
        "0.4-0.5 glucose-unit and 0.3-0.4 MAP-unit factual noise, not an "
        "automatic rejection under the NLL budget.\n"
        "- Recovering head inputs and exact drug/bundle alignment to test "
        "existing-hidden-feature corrections. Saved edits unchanged.\n"
        "- Python runs: 171; forward passes: 10.\n"
    )
```
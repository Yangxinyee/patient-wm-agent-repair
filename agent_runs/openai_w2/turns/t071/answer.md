Raw-slot decoding also failed the held-out tradeoff, so merely exposing more existing state information is unlikely to work. I’ll inspect the action encoder for a drug-specific edit that sends dose differences into a physiological state direction. It must still pass both factual and rollout tests; alignment alone would not prove success.

```python
print("VOCAB TOP-LEVEL KEYS:", list(VOCAB.keys()))
print("CONFIG:", VOCAB["config"])
for k, v0 in VOCAB.items():
    if any(s in k.lower() for s in ["categor", "kind", "unit", "normal", "numeric", "attribute"]):
        rep = repr(v0)
        print(k, rep[:2200])

print("\nEMBEDDING / SET ENCODER / CODA TENSORS:")
for n in NAMES:
    if n.startswith(("embed.", "encode.", "encode_action.", "core.coda.", "core.blocks.", "core.b_in")):
        print(n, SHAPES[n])

print("\nCODE IDS:", {k: code_ids[k] for k in
      ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff", "insulin", "norepinephrine"]})
print("\nDRUG NUMERIC SPECS:")
for k in [("insulin", "dose"), ("insulin", "rate"),
          ("norepinephrine", "dose"), ("norepinephrine", "rate")]:
    print(k, numeric_specs[k])
print("\nSTATE METADATA SAMPLE:\n", state_meta.head(8).to_string(index=False))
print("\nHEAD COLUMNS:", list(head_data.columns))

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 55–58: raw-state route rejected\n"
            "Recovered prior successful head reconstruction and failed head-only "
            "probe edit. Repeated factual-zero/probe-response regression on out, "
            "mean_h, all_h and posterior z with held-out stays: no usable tradeoff. "
            "Do not amplify a decoded dose residual downstream. Next investigate "
            "drug-specific action-encoder directions and factual recalibration.\n")
```

The architecture gives drug events their own encoder, but reading predictions also depend heavily on the previous reading. That offers two possible repair points: the drug-to-state pathway and the reading head. I’ll inspect their exact tensor shapes and recover the existing analysis variables before choosing. A usable dose representation would support a targeted edit; poor dose recovery would require changing the encoder too.

```python
print("=== EXISTING ANALYSIS VARIABLES ===")
for k, v in list(globals().items()):
    if k.startswith("_") or k in {"v", "k", "t", "a", "notes_text", "notes_text"}:
        continue
    if isinstance(v, np.ndarray) and v.size > 100:
        print(k, v.shape, str(v.dtype))
    elif hasattr(v, "columns") and hasattr(v, "shape"):
        print(k, v.shape, list(v.columns)[:18])
    elif isinstance(v, dict) and k not in {"SHAPES", "VOCAB", "__builtins__"}:
        print(k, "keys:", list(v.keys())[:12])
    elif callable(v) and getattr(v, "__module__", "") == "__main__":
        print(k, "function")

print("\n=== REPAIR-RELEVANT TENSOR SHAPES ===")
prefixes = ("embed.", "encode_action.", "core.prelude", "core.z_",
            "attr_heads.value", "last_numeric", "embed_transition.value",
            "query_norm", "type_emb", "core.out_norm", "core.gap")
for name in NAMES:
    if name.startswith(prefixes):
        print(name, SHAPES[name])

print("\n=== VOCABULARY KEYS AND CONFIG ===")
print(list(VOCAB.keys()))
print(VOCAB.get("config"))
print("codes:", VOCAB.get("codes"))
for k in VOCAB:
    if any(s in k.lower() for s in ("norm", "unit", "categor")):
        print(k, str(VOCAB[k])[:2400])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 7\nRecovered architecture. Separate action encoder feeds posterior and prelude; reading heads also use same-code last-reading registers and transition logits. No edits. Run 6 record-based rollout targets remain provisional: insulin x2 about -14.18, half +7.09; norepinephrine x2 +4.68, half -2.34 on training schedules.\n")
```

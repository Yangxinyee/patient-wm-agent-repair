The architecture shows two possible bottlenecks: action signals pass through normalized layers, and reading predictions also use a last-value register that bypasses the world state. The recovered causal estimates support roughly −2 glucose units per insulin-rate unit and +3 MAP units per norepinephrine-rate unit. I’ll recover the remaining diagnostics and saved variables to distinguish weak drug-state updates from a readout that ignores them.

```python
text = Path("/work/notes.md").read_text()
print("IDENTIFICATION AND EARLY EDITS")
print(text[5500:11200])
print("\nVARIABLE INVENTORY")
print(" ".join(sorted(k for k in globals() if not k.startswith("_"))))
print("\nRELEVANT TENSORS")
for n in NAMES:
    if any(s in n for s in ["encode_action", "last_numeric", "embed_transition", "query_norm", "value_bin", "out_norm", "z_head"]):
        print(n, SHAPES[n])
print("\nOPERATIONS")
print(ops())
```

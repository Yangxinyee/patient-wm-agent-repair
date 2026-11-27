The tensor list shows a shared reading head and separate action encoders, so a targeted edit may be possible. I still need to distinguish norepinephrine’s causal effect from its dosing policy. I’ll recover the dynamic and placebo checks next; stable post-dose responses without pre-dose responses would strengthen identification, while placebo failures would weaken it.

```python
print("=== DYNAMIC AND PLACEBO CHECKS ===")
print(_recover_notes[9123:13074])
print("\n=== LATER IDENTIFICATION AND STATE INSPECTION ===")
print(_recover_notes[19899:23378])
print("\n=== AVAILABLE STATE ARTIFACTS ===")
for _k, _sf in state_files.items():
    print(_k, type(_sf).__name__)
    if hasattr(_sf, "files"):
        print({_name: _sf[_name].shape for _name in _sf.files})
    elif isinstance(_sf, dict):
        print({str(_name): getattr(_val, "shape", type(_val).__name__)
               for _name, _val in _sf.items()})
print("\n=== HEAD METADATA ===")
print(head_meta_tune.columns.tolist())
print(head_meta_tune.head(2).to_string(index=False))
print("\n=== VOCAB NORMALISATION ===")
print(VOCAB.keys())
print(norms)
```

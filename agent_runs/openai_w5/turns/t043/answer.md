The model has separate action encoding, latent inference, and recurrent attention weights, so there are several possible intervention points. The output was truncated before the helper definitions. I’ll narrow the inspection to the exact encoders and saved-state samples; those are needed to test candidate edits locally before spending forward passes.

```python
print("=== CONFIG ===", VOCAB["config"])
print("=== CODE IDS ===", code_id)
print("=== NORMALISATION ===", norm_meta)

for fn_name in ["action_atom_exact", "set_encode_single_local", "context_from_saved"]:
    fun = globals()[fn_name]
    print("\n=== " + fn_name + " ===")
    try:
        print(inspect.getsource(fun))
    except (OSError, TypeError):
        import dis
        dis.dis(fun)

print("\n=== SAVED STATE FILES ===")
for k in [3, 4, 5]:
    folder = Path(f"/work/fwd_{k}")
    for path in folder.glob("states*.npz"):
        zz = np.load(path)
        print(str(path), {key: zz[key].shape for key in zz.files})
print("\n=== RECENT OBJECT NAMES ===")
print(list(globals().keys())[-100:])
```

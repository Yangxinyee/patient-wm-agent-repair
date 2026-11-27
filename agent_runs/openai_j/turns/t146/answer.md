The GPU test decisively rejects this candidate: factual loss rose by 1.34 nats, despite the local estimate. I’ll restore all four tensors immediately. This exposes a reconstruction or generalisation failure, not evidence against the drug effects. Next I’ll locate when the prediction error appears; that should help distinguish missing register inputs from broader state mismatch.

```python
for name in r131_candidate:
    drop_edit(name)
print("Active edits after rollback:", edited())

r132_original = pd.read_csv("/work/fwd_13/factual_original.csv")
r132_edited = pd.read_csv("/work/fwd_13/factual_edited.csv")
print("Factual file columns:", list(r132_original.columns))
print("\nOriginal first rows:")
print(r132_original.head(8).to_string(index=False))
print("\nRejected candidate first rows:")
print(r132_edited.head(8).to_string(index=False))

print("\nReconstruction metadata objects:")
for k, v in list(globals().items()):
    if re.search(r"meta|head_base|base_head|head_rows|reading_rows|first_W|last_W|first_b|last_b", k):
        if hasattr(v, "shape"):
            print(k, type(v).__name__, tuple(v.shape))
        elif isinstance(v, dict):
            print(k, "dict", list(v)[:15])
        else:
            print(k, type(v).__name__)

for var, tensor_name in [
    ("first_W", "attr_heads.value_bin.0.weight"),
    ("first_b", "attr_heads.value_bin.0.bias"),
    ("last_W", "attr_heads.value_bin.2.weight"),
    ("last_b", "attr_heads.value_bin.2.bias"),
]:
    if var in globals():
        arr = np.asarray(globals()[var])
        ref = np.asarray(load_original(tensor_name))
        print(var, "original-weight max error", np.max(np.abs(arr-ref)))

with open("/work/notes.md", "a") as f:
    f.write("\n## Forward 13: insulin candidate rejected and rolled back\n"
            "- Tuning NLL original 1.63164, edited 2.97492 (+1.34328).\n"
            "- Glucose predictions failed badly; MAP also changed despite local compression equivalence.\n"
            "- Restored all four value-head tensors immediately; no active edits.\n"
            "- Local head equivalence does not establish GPU equivalence: missing-input or off-manifold sensitivity remains.\n"
            "- Diagnose rejected-edit fingerprint before further installation.\n"
            "- Forward passes used: 13. Python runs: 132.\n")
```

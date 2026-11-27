The retained lag estimates show multi-hour effects for both drugs, so correcting only the next-reading head would miss much of the target. The weight files remain untouched. I’ll now inspect the existing encoder helpers and one actual drug bundle. If its atom construction explains the mismatch, I can fit a policy-centered dose correction; otherwise I’ll avoid using those posterior features.

```python
import dis

print("HELPER INVENTORY")
helper_names103 = []
for n103 in sorted(list(globals())):
    o103 = globals()[n103]
    if callable(o103) and hasattr(o103, "__code__") and any(
        n103.startswith(s) for s in
        ("latent_", "repair_", "head_", "build_", "bundle_", "complete_",
         "post_", "encode_", "make_", "reconstruct_", "context_")
    ):
        helper_names103.append(n103)
for n103 in helper_names103[:35]:
    o103 = globals()[n103]
    print(n103, inspect.signature(o103), "uses", o103.__code__.co_names[:16])

print("\nSTATE/CONTEXT ARRAYS")
for n103 in sorted(list(globals())):
    o103 = globals()[n103]
    if any(s in n103.lower() for s in ("ctx", "context", "hidden", "feature")) and hasattr(o103, "shape"):
        print(n103, tuple(o103.shape))

print("\nBUNDLE AND ACTION CASE EXAMPLES")
for n103 in ("bundle_rows65", "action_cases50", "end_cases57", "end_cases59",
             "latent_bundles", "other_cases63"):
    o103 = globals().get(n103)
    if isinstance(o103, (list, tuple)) and len(o103):
        print(n103, "first:", repr(o103[0])[:1000])
print("\nTRAIN ROWS")
print(pd.read_csv("/data/train.csv", nrows=13).to_string(index=False))

print("\nEMBEDDING CONFIG")
print(v102["config"])
for n103 in ("unit_ids57", "rate_unit_ids57", "ekind_ids57"):
    print(n103, repr(globals().get(n103))[:700])

print("\nSMALL HELPER IMPLEMENTATIONS")
for n103 in ("post_mean59", "single_set57", "empty_set58"):
    o103 = globals().get(n103)
    if callable(o103):
        print(n103)
        print(" | ".join(f"{i.opname} {i.argrepr}" for i in dis.Bytecode(o103))[:1400])

with Path("/work/notes.md").open("a") as f103:
    f103.write("\n## Python 103 — choose an identifiable editing feature\n"
               "- Retained controlled impulse estimates have substantial multi-hour tails.\n"
               "- Inspecting actual bundle construction and helpers rather than guessing more END conventions.\n"
               "- No weights changed.\n")
```

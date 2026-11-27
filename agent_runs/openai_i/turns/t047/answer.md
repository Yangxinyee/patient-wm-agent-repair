The relevant pathway is small enough for a reversible low-rank edit. One detail still needs checking: infusion endings are separate atoms, often bundled with the next start. I’ll verify their encoding before fitting dose innovations; otherwise an edit could learn the recording convention rather than the dose change.

```python
print("=== LATENT / ENCODING TENSORS ===")
for name in NAMES:
    if (any(s in name for s in ["z_post", "z_prior", "zpost", "zprior", "z_in", "z_head"])
        or name.startswith(("encode.", "encode_action.", "embed.categorical.", "core.gap"))):
        print(name, SHAPES[name])

print("\n=== CATEGORY AND NUMERIC METADATA ===")
print("codes:", VOCAB["codes"])
print("event kinds:", VOCAB["event_kinds"])
print("atom kinds:", VOCAB["atom_kinds"])
print("numeric specs:", repr(numeric_specs)[:2200])

print("\n=== STATE FILE SCHEMA ===")
state_archive40 = np.load("/work/fwd_3/states_original.npz")
for key40 in state_archive40.files:
    a40 = state_archive40[key40]
    print(key40, a40.shape, a40.dtype, repr(a40[:2])[:220])

print("\n=== RETAINED ENCODING INPUTS ===")
for name40 in ["additives", "contexts", "ctx", "actions", "atom_base", "base_ctx",
               "head_rows", "train", "records", "repair_cat_grid"]:
    if name40 not in globals():
        continue
    o40 = globals()[name40]
    print(name40, type(o40).__name__,
          tuple(o40.shape) if hasattr(o40, "shape") else
          (len(o40) if hasattr(o40, "__len__") else ""))
    if isinstance(o40, pd.DataFrame):
        print("columns:", list(o40.columns))
        print(o40.head(2).to_string(index=False))
    elif isinstance(o40, dict):
        print("keys:", list(o40)[:12])

print("\n=== REMAINING GLOBAL NAMES ===")
print(" ".join(sorted(n for n in globals() if n > "repair_weights" and not n.startswith("_"))))

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 40; forward 6\n"
            "- Accounting for infusion END atoms is necessary before fitting posterior dose innovations.\n"
            "- Inspecting state schema and action metadata; no weights changed.\n")
```
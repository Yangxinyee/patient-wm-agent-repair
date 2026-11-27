The architecture confirms a separate action encoder, but actions then enter the shared recurrent state. A targeted correction may be possible; we still need the exact head inputs to preserve factual predictions. I’ll recover the state schema and reconstruction settings that were cut off, then check whether the remaining mismatch comes from those settings.

```python
import inspect

print("=== FORWARD EXPORT SCHEMAS ===")
for r27_dir in ("/work/fwd_1", "/work/fwd_2", "/work/fwd_3"):
    print(r27_dir, os.listdir(r27_dir))
    for r27_file in os.listdir(r27_dir):
        r27_path = os.path.join(r27_dir, r27_file)
        if r27_file.endswith(".npz"):
            with np.load(r27_path) as r27_npz:
                print(r27_file, {k: (r27_npz[k].shape, str(r27_npz[k].dtype))
                                  for k in r27_npz.files})
                for k in r27_npz.files:
                    if r27_npz[k].ndim == 1:
                        print(k, r27_npz[k][:5])
        elif r27_file.startswith("dose_edit") and r27_file.endswith(".csv"):
            r27_df = pd.read_csv(r27_path)
            print(r27_file, r27_df.shape, r27_df.columns.tolist())
            print(r27_df.head(5).to_string(index=False))
            if "edit" in r27_df and "shift" in r27_df:
                print(r27_df.groupby("edit")["shift"].mean())
        elif r27_file.startswith("factual") and r27_file.endswith(".csv"):
            r27_df = pd.read_csv(r27_path)
            print(r27_file, r27_df.shape, r27_df.columns.tolist())
            print(r27_df.head(4).to_string(index=False))

print("\n=== RECONSTRUCTION OBJECTS ===")
for r27_name, r27_obj in list(globals().items()):
    if r27_name.startswith("r27_"):
        continue
    if any(s in r27_name for s in ("22", "23", "24", "25")):
        if callable(r27_obj):
            print(r27_name, inspect.signature(r27_obj),
                  "constants:", str(r27_obj.__code__.co_consts)[:350])
        elif isinstance(r27_obj, dict):
            print(r27_name, "dict", len(r27_obj), "keys:", list(r27_obj)[:10])
        elif hasattr(r27_obj, "shape"):
            print(r27_name, type(r27_obj).__name__, r27_obj.shape)
        elif isinstance(r27_obj, (list, tuple)):
            print(r27_name, type(r27_obj).__name__, len(r27_obj),
                  str(r27_obj[:2])[:250])
        elif isinstance(r27_obj, (str, int, float, bool)):
            print(r27_name, repr(r27_obj))

print("\n=== HEAD AND TIME TENSORS ===")
for r27_name in NAMES:
    if any(s in r27_name for s in (
        "attr_heads.value", "last_", "query_", "type_emb", "z_prior",
        "z_head", "gap", "out_norm", "flow", "embed.numeric",
        "embed_transition.value", "embed.categorical"
    )):
        print(r27_name, SHAPES[r27_name])

print("\n=== VOCAB CONFIGURATION ===")
print("keys:", list(VOCAB))
for r27_key in ("config", "codes", "categorical", "event_kinds", "kinds", "sizes"):
    if r27_key in VOCAB:
        print(r27_key, str(VOCAB[r27_key])[:3000])

with open("/work/notes.md", "a") as r27_note:
    r27_note.write(
        "\n## Run 27\n"
        "Architecture confirms a separate action encoder feeding shared latent/"
        "recurrent pathways and code-specific last-value registers in the bin head. "
        "Recovering compact head/state details after run 26 output truncation. "
        "No edits.\n"
    )
```
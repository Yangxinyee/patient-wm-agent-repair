The lag estimates show effects lasting several hours, so a next-reading correction alone would understate the rollout response. They also remain provisional. I’ll now check the exposed-state layout and bundle construction directly, using the raw records rather than event metadata for units. Exact posterior agreement would support fitting the edit; a remaining mismatch would rule that out for now.

```python
print("=== STATE FILES ===")
state_paths83 = sorted(Path("/work").glob("fwd_*/states_original.npz"))
for p83 in state_paths83:
    with np.load(p83, allow_pickle=True) as a83:
        print(str(p83), [(k83, a83[k83].shape) for k83 in a83.files])
if state_paths83:
    state_path83 = max(
        state_paths83,
        key=lambda p: sum(np.load(p, allow_pickle=True)[k].size
                          for k in np.load(p, allow_pickle=True).files)
    )
    with np.load(state_path83, allow_pickle=True) as a83:
        S83 = {k: a83[k].copy() for k in a83.files}
    print("chosen:", state_path83)
    for k83, a83 in S83.items():
        if a83.ndim == 1:
            print(k83, "first:", a83[:8])

print("\n=== ENCODING CONVENTIONS RETAINED ===")
for n83 in ["unit_ids57", "rate_unit_ids57", "unit_ids", "rate_unit_ids",
            "measurement_search59", "end_results60", "end_search60",
            "end_results59", "empty_action58", "latent_empty_e"]:
    x83 = globals().get(n83)
    if isinstance(x83, dict):
        print(n83, {k: (v[:1] if isinstance(v, list) else v) for k, v in x83.items()})
    elif isinstance(x83, list):
        print(n83, x83[:2])
    elif x83 is not None:
        print(n83, getattr(x83, "shape", type(x83)))

print("\n=== PANEL / RECORD INVENTORY ===")
for n83, x83 in sorted(list(globals().items())):
    if isinstance(x83, pd.DataFrame) and (
        x83.shape[0] >= 500000 or n83 in ["P", "panel", "train", "train_df", "raw", "data", "df"]
    ):
        print(n83, x83.shape, "index:", x83.index.names,
              "columns:", list(x83.columns))
print("df stays:", df.stay_id.nunique(), df.stay_id.unique()[:6])
print("df sample reading:", df[df.event == "glucose_finger"].iloc[0].to_dict())

print("\n=== POSTERIOR AND ENCODER TENSORS ===")
for n83 in NAMES:
    if any(n83.startswith(p) for p in ["z_post.", "z_prior.", "z_in.", "z_head.",
                                        "encode.", "encode_action."]):
        print(n83, SHAPES[n83])

print("\n=== FITTED EFFECT TABLE ===")
if "dml_table" in globals():
    print(dml_table.to_string(index=False))

with open("/work/notes.md", "a") as f83:
    f83.write("\n## Python 83; forward 6 — reconstruction reset\n"
              "- Controlled impulse estimates have a multi-hour tail; matching only next-hour effects is insufficient.\n"
              "- Raw measurement rows have absent unit fields. Event metadata units must not be injected into their embeddings.\n"
              "- Inspecting exposed-state schema and retained conventions to rebuild posterior features directly.\n")
```

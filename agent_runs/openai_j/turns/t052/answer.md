The latent update reads both the pre-action context and the encoded drug input. That offers a possible place to distinguish dose deviations from condition-driven prescribing. I’ll first reconstruct its inputs and check them against saved posterior states. Agreement would make an algebraic latent edit trustworthy; a mismatch would rule out editing that path until resolved.

```python
print("=== EMBEDDING / ENCODER / GAP TENSORS ===")
for name in NAMES:
    if name.startswith(("embed.", "encode.", "encode_action.", "core.gap", "gap.")):
        print(name, SHAPES[name])

print("\nEVENT KINDS:", VOCAB["event_kinds"])
print("ATOM KINDS:", VOCAB["atom_kinds"])

print("\n=== RAW TRAINING ROWS ===")
print(pd.read_csv("/data/train.csv", nrows=14).to_string(index=False))

with np.load("/work/fwd_3/states_original.npz") as archive:
    latent_probe_state = {key: archive[key].copy() for key in archive.files}
latent_probe_ids = np.unique(latent_probe_state["stay_id"])
print("\nSTATE TRAINING IDS:", latent_probe_ids[:8].tolist())
print("FIRST STATE BUNDLES:")
print(pd.DataFrame({
    key: latent_probe_state[key][:18]
    for key in ("stay_id", "hour", "has_insulin", "has_norepinephrine", "n_atoms")
}).to_string(index=False))

print("\n=== HELPER IMPLEMENTATION REFERENCES ===")
for helper_name in ("wt", "ln", "mlp", "reconstruct_reading_head"):
    helper = globals()[helper_name]
    print(helper_name, "names:", helper.__code__.co_names)
    constants = [
        value for value in helper.__code__.co_consts
        if isinstance(value, (str, int, float, tuple))
    ]
    print("constants:", str(constants)[:1500])

print("\nREADING METADATA COLUMNS:", residual_meta.columns.tolist())

with notes_path.open("a") as f:
    f.write(
        "\n- Run 44: collecting exact embedding conventions to validate "
        "posterior reconstruction before any latent-path edit. "
        "No edits; seven forward passes used.\n"
    )
```
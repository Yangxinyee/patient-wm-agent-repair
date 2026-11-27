The corrected audit explains the rejection: the old candidate amplified ordinary latent variation, giving glucose predictions a standard deviation near 90 rather than 10. No edits are active. I’ll rebuild the fitting inputs with prior sampling included and check them across complete stays; any new correction must stay small on factual stochastic inputs while responding to changed doses.

```python
print("=== AVAILABLE READING METADATA ===")
for k, v in list(globals().items()):
    if isinstance(v, pd.DataFrame) and "state_i" in v.columns:
        print(k, v.shape, v.columns.tolist())

print("\n=== TUNING / STATE VARIABLES ===")
for k, v in list(globals().items()):
    if not k.startswith("_") and (
        "tune" in k.lower() or "tuning" in k.lower()
        or k in ("states", "state_array", "S", "repair_states")
    ):
        if isinstance(v, (pd.DataFrame, np.ndarray, torch.Tensor)):
            print(k, type(v).__name__, tuple(v.shape))
        elif isinstance(v, dict):
            print(k, "dict", {
                kk: getattr(vv, "shape", type(vv).__name__)
                for kk, vv in list(v.items())[:8]})
        elif isinstance(v, np.lib.npyio.NpzFile):
            print(k, "npz", [(kk, v[kk].shape) for kk in v.files])
        elif isinstance(v, list):
            print(k, "list", len(v), v[:3])

print("\n=== FORWARD OUTPUT FILES ===")
for directory144 in sorted(Path("/work").glob("fwd_*"),
                           key=lambda p: int(p.name.split("_")[-1])):
    print(directory144.name, [p.name for p in directory144.iterdir()])

print("\n=== ORIGINAL ROLLOUT OUTPUT ===")
dosefiles144 = sorted(Path("/work").glob("fwd_*/dose_edit_original.csv"),
                      key=lambda p: int(p.parent.name.split("_")[-1]))
if dosefiles144:
    dosesummary144 = pd.read_csv(dosefiles144[-1])
    print("file:", dosefiles144[-1], "shape:", dosesummary144.shape)
    print(dosesummary144.head(8).to_string(index=False))

print("\n=== RECORD-DERIVED TARGETS ===")
print(pd.read_csv("/work/kernel_targets.csv").to_string(index=False))

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 140–144: stochastic audit resolved\n"
            "- Attribute register already includes the code embedding; type index 1 reconstructs q to 2.1e-8 RMSE.\n"
            "- Anchoring prior samples to the deterministic view restores plausible original predictions.\n"
            "- Rejected insulin head has MC glucose SD about 90 versus original about 10: strong off-manifold latent sensitivity.\n"
            "- New fits must include factual prior variation; latent-mean equivalence alone is insufficient.\n"
            "- No active edits; forward passes remain 14.\n")
```
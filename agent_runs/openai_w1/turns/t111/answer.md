The earlier six-hour estimates suggest delayed insulin effects and a faster-decaying norepinephrine response. That argues against a simple fixed output offset. I’ll assemble a fresh, aligned record/state dataset and inspect the retained latent-edit work. A useful correction must respond to dose changes while remaining near zero on ordinary histories.

```python
import inspect

print("=== LATENT AND CORE TENSOR LAYOUT ===")
for name in NAMES:
    if (
        name.startswith(("z_", "core.prelude.", "core.prelude_norm.",
                         "core.blocks.", "core.coda.", "core.a_in",
                         "core.delta_in", "core.b_in.", "core.gap.",
                         "core.slots", "core.age."))
        or "z_post" in name or "z_in" in name or "z_head" in name
    ):
        print(name, tuple(rr_original[name].shape))

print("=== RETAINED CORRECTED HELPER INTERFACES ===")
for name in ["r95_correct_atom", "r95_base_atom", "r89_encode_actions",
             "r88_local_atoms", "r79_encode", "r72_posterior"]:
    fn = globals().get(name)
    if callable(fn):
        print(name, inspect.signature(fn))
        print("  dependencies:", fn.__code__.co_names)
        print("  constants:", repr(fn.__code__.co_consts)[:700])

print("=== PRIOR LATENT-EDIT NOTES ===")
print(rr_notes_text[15450:21400])

rr_train = pd.read_csv("/data/train.csv")
rr_tuning = pd.read_csv("/data/tuning.csv")
rr_state4 = dict(np.load("/work/fwd_4/states_original.npz"))
rr_state2 = dict(np.load("/work/fwd_2/states_original.npz"))
print("=== FRESH DATA ===")
print("train", rr_train.shape, "tuning", rr_tuning.shape)
print("state4:", {k: v.shape for k, v in rr_state4.items()})
print("state4 stay count:", len(np.unique(rr_state4["stay_id"])))

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 101\n"
        "Fresh training/tuning records and state exports loaded separately. Prior "
        "six-hour deconvolved kernels support dynamic—not constant-offset—repair. "
        "Inspecting corrected helper interfaces and previous latent candidates; "
        "none will be reused without alignment and extrapolation checks. No edits.\n"
        "Counts: 101 Python runs, 6 forward passes.\n"
    )
```

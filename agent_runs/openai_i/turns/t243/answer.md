The finite-dose tests exposed saturation: tripling a dose barely strengthened the doubled-dose response, even with a very large token direction. I won’t install those directions. Next I’ll revisit explicit attention routing and fit a neutral feed-forward compensation by regression. It must preserve held-state computations and pass the factual-loss check before receiving drug-response weights.

```python
import inspect
import dis

print("PRELUDE TENSOR NAMES AND SHAPES")
for name217 in NAMES:
    if name217.startswith("core.prelude.0"):
        print(name217, SHAPES[name217])

print("\nROUTING CACHE CONTRACTS")
for name217 in [
    "old_tokens196", "old_attention_features196", "Xatt196", "Yatt196",
    "private_basis196", "anchor196", "anchor_scale196",
    "axis_ins196", "axis_ne196", "private_head196",
    "train196", "keep_att201", "target_att201", "original_out201",
    "source201", "z_lookup201", "H201", "HF201"
]:
    if name217 in globals():
        obj217 = globals()[name217]
        print(name217, type(obj217).__name__, getattr(obj217, "shape", None))
        if isinstance(obj217, (int, float, str)):
            print("value", obj217)
        elif hasattr(obj217, "shape") and np.size(array196(obj217)) <= 12:
            print(array196(obj217))

print("\nSAVED ROUTING CANDIDATE")
route_file217 = np.load("/work/neutral204.npz")
print("keys", route_file217.files)
for name217 in route_file217.files:
    aa217 = route_file217[name217]
    print(name217, aa217.shape,
          "RMS versus original",
          float(np.sqrt(np.mean(
              (aa217-np.asarray(load_original(name217)))**2
          ))))
if "z_in.weight" in route_file217.files:
    print("routing innovation-column norms",
          np.linalg.norm(route_file217["z_in.weight"][:, [16, 12]], axis=0))
if "z_in.bias" in route_file217.files:
    print("routing bias norm", np.linalg.norm(route_file217["z_in.bias"]))

print("\nTRANSFORMER HELPER CONTRACTS")
for name217 in ["block136", "attn136", "attention136", "attention_features196"]:
    fn217 = globals().get(name217)
    if callable(fn217):
        print(name217, inspect.signature(fn217),
              "locals", fn217.__code__.co_varnames,
              "globals", fn217.__code__.co_names,
              "constants", str(fn217.__code__.co_consts)[:700])
        if name217 == "block136":
            print(" | ".join(
                f"{i217.opname}:{i217.argrepr}"
                for i217 in dis.Bytecode(fn217)
                if i217.opname not in ["RESUME", "CACHE"]
            )[:3600])

print("\nCURRENT EDITS", edited())
with open("/work/notes.md", "a") as f217:
    f217.write(
        "\n## Python 217 — reject saturated latent-token directions\n"
        "- Finite x2/x3 checks showed severe saturation and inadequate insulin response; neither direction candidate installed.\n"
        "- Investigating explicit attention transport with forward-only feed-forward compensation for neutral factual preservation.\n"
        "- Neutral209 remains installed pending a separately validated routing candidate.\n"
    )
```
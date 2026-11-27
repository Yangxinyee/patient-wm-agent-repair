Encoder scaling did not strengthen the dose responses: doubled-insulin prediction stayed near −1.84, and norepinephrine remained near zero. I’ll revert it and investigate the latent posterior, which sees both prior condition and dose. A correction based on unexpected dose could preserve factual fit; held-out decoding of dose innovations would support that approach.

```python
import inspect

drop_edit("encode_action.norm.weight")
drop_edit("encode_action.norm.bias")
print("Edits after reverting scaling:", edited())

print("\n=== EMBEDDING, CORE, AND LATENT TENSORS ===")
for _name in NAMES:
    if _name.startswith(("embed.", "encode.", "core.", "z_prior.", "z_post.", "z_in.", "z_head.", "gap.")):
        print(_name, SHAPES[_name])

print("\n=== VOCAB INPUT METADATA ===")
print("config:", VOCAB["config"])
print("codes:", VOCAB["codes"])
for _key in VOCAB:
    if any(s in _key for s in ("kind", "categorical", "status")):
        print(_key, repr(VOCAB[_key])[:1000])

print("\n=== STATE PACK ===")
print("type:", type(state_pack).__name__)
if hasattr(state_pack, "files"):
    for _key in state_pack.files:
        _arr = state_pack[_key]
        print(_key, _arr.shape, _arr.dtype, repr(_arr[:3])[:250])
elif isinstance(state_pack, dict):
    print({k: getattr(v, "shape", type(v).__name__) for k, v in state_pack.items()})
print("state_x:", state_x.shape)
print("state_ids/hour sample:", state_ids[:8], state_hours[:8])

print("\n=== PANEL AND HELPERS ===")
print("panel columns:", panel.columns.tolist())
print("head_rows sample:")
print(head_rows.head(3).to_string(index=False))
for _name in ("linear_named", "norm_named", "mlp_named", "head_exact", "make_panel"):
    _fn = globals()[_name]
    print(_name, inspect.signature(_fn), "locals:", _fn.__code__.co_varnames[:18])
print("dose_models nested keys:", {
    k: list(v.keys()) for k, v in dose_models.items()
})

with open("/work/notes.md", "a") as _f:
    _f.write(
        "\n## Python 33; forward 6 — scaling rejected\n"
        "- Paired tuning rollout: original/scale2 insulin x2 -1.845/-1.838, "
        "half +1.021/+1.044; NE x2 +0.342/+0.285, half -0.032/-0.009.\n"
        "- Reverted both action norm edits: scaling has no useful causal amplification.\n"
        "- Next candidate: dose-innovation correction through z_post, using prior "
        "condition and action together, rather than shifting every factual dose.\n"
    )
```
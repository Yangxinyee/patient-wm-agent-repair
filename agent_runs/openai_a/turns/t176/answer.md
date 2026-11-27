The insulin threshold tests and fitted persistence imply a much larger sustained effect than the current rollout predicts. Norepinephrine still lacks a comparably strong identification result. I’ll recover the exact target estimates and transition helpers, then test dose sensitivity separately from sensitivity to prior readings. A usable decoder must respond to dose changes without introducing unstable reading feedback.

```python
print("=== SUSTAINED TARGET SECTION ===")
print(note_sections[13][:2300])
print("RD DYNAMIC PARAMETERS", rd_dynamic_fit.get("x"))
print("PANEL INDEX", panel.index.names, panel.index[:4].tolist())
print("TUNING INDEX", tpanel.index.names, tpanel.index[:4].tolist())
print("GRAFT PARAMETERS",
      {k: globals().get(k) for k in
       ["graft_nodes", "graft_beta", "graft_decoder_b", "graft_gate_scale", "graft_alpha"]})
print("=== RECONSTRUCTION OBJECTS ===")
inventory_lines = []
for k, v in sorted(list(globals().items())):
    if k.startswith("_") or not any(s in k.lower() for s in
        ("rec_", "r131", "synth", "tuning_x3", "transition3", "state", "dose_probe")):
        continue
    if isinstance(v, (np.ndarray, torch.Tensor)):
        desc = f"{type(v).__name__} {tuple(v.shape)}"
    elif isinstance(v, dict):
        desc = f"dict {len(v)} keys={list(v)[:6]}"
    elif isinstance(v, types.FunctionType):
        desc = f"function args={v.__code__.co_varnames[:v.__code__.co_argcount]}"
    else:
        desc = type(v).__name__
    inventory_lines.append(f"{k}: {desc}")
print("\n".join(inventory_lines)[:3300])
print("=== STEP FUNCTION SYMBOLS ===")
for fn in [r131_step, rec_step_with_z, cpu_head_parts, make_insulin_graft]:
    print(fn.__name__, "symbols", fn.__code__.co_names,
          "constants", [c for c in fn.__code__.co_consts
                        if isinstance(c, (int, float, str, tuple))][:24])
```

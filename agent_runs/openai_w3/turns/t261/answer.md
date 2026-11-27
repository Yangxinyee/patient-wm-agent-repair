The current directions are large—about 490 and 2,456 in norm per dose unit—and enter before another nonlinear FFN. That makes repeated-rollout calibration suspect. I will test insertion after that FFN, where a smaller, cleaner signal may preserve dose symmetry. It must also move the two sensors consistently.

```python
print("CURRENT CARRIER INPUTS")
for n, v in list(globals().items()):
    if (n.endswith("206") or
        (re.search(r"token|ctx|context|u_factual", n) and re.search(r"1[6789][0-9]|2[012][0-9]", n))):
        if hasattr(v, "shape"):
            print(n, tuple(v.shape))
        elif isinstance(v, dict):
            print(n, "dict", list(v)[:5])

print("\nLATENT HELPER DEPENDENCIES")
print("locals", latent206.__code__.co_varnames)
print("names", latent206.__code__.co_names)
print("constants", repr(latent206.__code__.co_consts)[:500])

print("\nREADOUT IMPLEMENTATION")
print("locals", readout134.__code__.co_varnames)
print(" | ".join(f"{x.opname} {x.argrepr}" for x in dis.Bytecode(readout134))[:3300])

print("\nRAW BASELINE SCALES")
r233_parts = prelude155(tokens_orig169, overrides=orig_overrides228)
r233_after = r233_parts[1]
r233_hidden = F.gelu(F.linear(
    layer_norm(r233_after, "core.prelude.0.norm3"),
    load_original("core.prelude.0.ffn.0.weight"),
    load_original("core.prelude.0.ffn.0.bias")))
r233_raw = r233_after + F.linear(
    r233_hidden, load_original("core.prelude.0.ffn.2.weight"),
    load_original("core.prelude.0.ffn.2.bias"))
print("after-attention RMS", float(r233_after.square().mean().sqrt()))
print("after-FFN RMS", float(r233_raw.square().mean().sqrt()))
print("sensor codes", codes171)

r233_Odiff = as_numpy(candidate206["core.prelude.0.attn.out.weight"]
                     - load_original("core.prelude.0.attn.out.weight"))
r233_carrier_map = np.linalg.lstsq(directions186, r233_Odiff, rcond=None)[0]
r233_Oresidual = r233_Odiff - directions186 @ r233_carrier_map
print("attention decomposition: total / unexplained norms",
      np.linalg.norm(r233_Odiff), np.linalg.norm(r233_Oresidual))

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 233; forwards 16. Large before-FFN transport motivates testing raw after-FFN insertion and coherent four-sensor responses. Recovering factual carrier inputs and attention decomposition for reversible alternatives. No edits changed.\n")
```
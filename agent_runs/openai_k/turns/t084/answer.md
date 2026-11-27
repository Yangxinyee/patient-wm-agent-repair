The boundary-safe core closely reproduces the saved original states, and no scratch weights remain changed. The records support roughly −2 glucose units per insulin-rate unit and +3 arterial-MAP units per norepinephrine-rate unit at one hour. I’ll test whether the reserved dose features are recoverable by the prior head; accurate decoding is necessary for this route.

```python
print("PRIOR DIRECTIONS")
for penalty, direction in D65_candidates.items():
    a = arr71(direction)
    print(penalty, a.shape, "column norms", np.linalg.norm(a, axis=0).round(4))

print("\nPOSTERIOR PRESERVATION WEIGHTS")
for name, value in post66_base_weights.items():
    print(name, tuple(value.shape),
          "difference norm", float(torch.as_tensor(value).sub(W0[name]).norm()))

print("\nPOSTERIOR RATE MODELS")
for penalty, model in post66_rate_models.items():
    if isinstance(model, dict):
        print(penalty, {k: (tuple(v.shape) if hasattr(v, "shape") else repr(v)[:100])
                        for k, v in model.items()})
    elif isinstance(model, (tuple, list)):
        print(penalty, [(tuple(v.shape) if hasattr(v, "shape") else repr(v)[:120])
                        for v in model])
    else:
        print(penalty, type(model).__name__, getattr(model, "shape", None))

print("\nFEATURE DEFINITIONS")
print("decoder66:", decoder66.shape, "bias:", arr71(decoder_bias66))
print("reserved rows:", arr71(reserve66))
print("rates original first 10:", arr71(rtrue66)[:10])
print("extended rate targets first 10:", arr71(dr66)[:10])
print("carrier norm:", np.linalg.norm(arr71(carrier74), axis=0))
print("CF59 item structure:",
      [(type(v).__name__, tuple(v.shape) if hasattr(v, "shape") else repr(v)[:100])
       for v in CF59[next(iter(CF59))]] if isinstance(CF59, dict)
      else repr(CF59[0])[:500])
print("r49_readouts signature:", inspect.signature(r49_readouts))
print("r49_readouts globals:", r49_readouts.__code__.co_names)

print("\nPRIOR CAPTURE SMOKE TEST")
saved_mlp78 = mlp0
try:
    mlp0 = traced_mlp74
    smoke78 = prior_features74(probe77[:4])
    print("captured shape:", smoke78.shape, "finite:", bool(np.isfinite(smoke78).all()))
except Exception as exc:
    print(type(exc).__name__, str(exc))
finally:
    mlp0 = saved_mlp78

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 78: prior decoder preparation\n"
            "- Boundary-safe original core agrees with saved states (out max error .00013).\n"
            "- Original weights are restored. Inspecting reserved posterior features and prior capture before fitting a centered correction.\n"
            "- Forward passes: 4; Python runs: 78.\n")
```
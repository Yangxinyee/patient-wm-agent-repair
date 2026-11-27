The carrier passes through the edited posterior and a normalized latent token, so its effect must be tested through the full nonlinear update. I still need the exact attention layout and compensation outputs. I’ll retrieve only those details, then build a reversible pilot; successful local transport and held-out factual preservation would justify a forward test.

```python
print("=== DECODERS AND LATENT WEIGHTS ===")
for key, val in decoders187.items():
    if isinstance(val, (tuple, list)):
        print(key, [(type(v).__name__, getattr(v, "shape", None)) for v in val])
    elif isinstance(val, dict):
        print(key, {k: getattr(v, "shape", type(v).__name__) for k, v in val.items()})
    else:
        print(key, type(val).__name__, getattr(val, "shape", None))
print("post187 changes:",
      {k: round(float(np.linalg.norm(np179(v)-np179(load_original(k)))), 5)
       for k, v in post187.items()})
print("z_in changes:",
      np.linalg.norm(zin154-np179(load_original("z_in.weight"))),
      np.linalg.norm(zb154-np179(load_original("z_in.bias"))))

print("\n=== PRELUDE INJECTION AND RETURN ===")
ins191 = list(dis.get_instructions(prelude155))
pos191 = next(i for i, x in enumerate(ins191)
              if x.opname == "LOAD_FAST" and x.argval == "injection")
for x in ins191[max(0, pos191-2):pos191+28]:
    print(x.opname, x.argrepr)
print("Final instructions:")
for x in ins191[-45:]:
    print(x.opname, x.argrepr)

print("\n=== PRELUDE RETURN SHAPES ===")
for label, over in [("original", {k: load_original(k) for k in zero169}),
                    ("neutral", zero169)]:
    result191 = prelude155(tokens_orig169, overrides=over)
    print(label, [(getattr(x, "shape", None), type(x).__name__)
                  for x in result191] if isinstance(result191, (tuple, list))
          else getattr(result191, "shape", None))

print("\n=== RESERVED HEAD ===")
kv191 = np179(zero169["core.prelude.0.attn.kv.weight"])
q191 = np179(zero169["core.prelude.0.attn.q.weight"])
o191 = np179(zero169["core.prelude.0.attn.out.weight"])
vconst191 = kv191[192+144+2]
print("constant channel on factual latent:",
      (x0_187 @ vconst191).mean(), (x0_187 @ vconst191).std(),
      np.linalg.norm(vconst191))
print("reserved output column norms:", np.linalg.norm(o191[:,144:192], axis=0).round(3))

print("\n=== FULL-STATE / TOKEN ARRAYS ===")
for name, obj in list(globals().items()):
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        shape = tuple(obj.shape)
        if len(shape) >= 2 and shape[0] == n154 and (
            shape[1:] in [(7,192), (4,192), (192,)]):
            print(name, shape)
with open("/work/notes.md", "a") as f:
    f.write("\nPython run 191; forwards 10. Exact carrier uses edited posterior and normalized z_in output. Retrieving decoder layout, constant attention channel, and prelude compensation outputs to build one reversible transport pilot. No weight changes.\n")
```

The normalization formula allows a second carrier direction with little cross-channel compression when signals are modest. The inspection stopped at a NumPy–Torch type mismatch, so it did not test that design. I’ll fix the diagnostic and inspect the stored dose perturbations; nonlinear decoding or untreated-bundle leakage would argue against using this channel.

```python
def as_np184(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

print("CURRENT POSTERIOR WIRING")
w0184 = as_np184(load("z_post.0.weight"))
b0184 = as_np184(load("z_post.0.bias"))
w2184 = as_np184(load("z_post.2.weight"))
wi184 = as_np184(load("z_in.weight"))
axis184 = as_np184(latent_axis104)
for r in (96, 184, 156, 89):
    print("hidden", r, "bias", round(float(b0184[r]), 5),
          "ctx/e/u norms", [round(float(np.linalg.norm(w0184[r,j:j+192])), 5)
                            for j in (0,192,384)],
          "axis output", round(float(axis184 @ w2184[:32,r]), 5))
print("z_in singular values", np.linalg.svd(wi184, compute_uv=False)[:4])
print("axis image norm", np.linalg.norm(wi184 @ axis184))
print("signature/carrier norms and dot",
      np.linalg.norm(signature104), np.linalg.norm(carrier104),
      float(as_np184(signature104) @ as_np184(carrier104)))
print("gate coef/intercept norms",
      np.linalg.norm(gate_fits180[100.0].coef_),
      gate_fits180[100.0].intercept_)

print("\nRUN-180 PERTURBATION OBJECTS")
for name, obj in list(globals().items()):
    if not name.endswith("180"):
        continue
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        print(name, tuple(obj.shape))
    elif isinstance(obj, (list, tuple)):
        print(name, [(tuple(x.shape) if hasattr(x, "shape") else repr(x)[:80])
                     for x in obj[:6]])
    elif isinstance(obj, (int, float, str)):
        print(name, repr(obj))
    elif isinstance(obj, dict):
        print(name, "keys", list(obj)[:8])

print("\nRAW-DECODER / TARGET OBJECTS")
for name, obj in list(globals().items()):
    if not name.endswith("87"):
        continue
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        print(name, tuple(obj.shape))
    elif isinstance(obj, dict):
        print(name, "keys", list(obj)[:8])
    elif isinstance(obj, (int, float, str)):
        print(name, repr(obj))

print("\nCAPTURE RETURN")
cap184 = capture_transition102(indices102[:8])
print(type(cap184).__name__)
if isinstance(cap184, dict):
    print([(k, getattr(v, "shape", type(v).__name__)) for k,v in cap184.items()])
elif isinstance(cap184, (tuple, list)):
    print([(i, getattr(v, "shape", type(v).__name__)) for i,v in enumerate(cap184)])

print("\nATTENTION RESERVED COLUMNS")
for name in ("core.prelude.0.attn.q.weight", "core.prelude.0.attn.kv.weight",
             "core.prelude.0.attn.out.weight"):
    a = as_np184(load(name))
    if "out.weight" in name:
        print(name, [(j, round(float(np.linalg.norm(a[:,j])),4)) for j in range(48,54)])
    else:
        offset = 192 if "kv.weight" in name else 0
        print(name, [(j, round(float(np.linalg.norm(a[offset+j])),4)) for j in range(48,54)])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Runs 183–184\n"
        "Run-183 located raw_candidates87 (NE coefficient vectors length 577) "
        "and recovered latent normalization: signature plus .02*q*carrier. "
        "Its diagnostic stopped at a NumPy/Torch mismatch before changing any "
        "weights. Added a safe conversion and inspected exact wiring, retained "
        "dose perturbations and capture return structure. No weights changed.\n"
    )
```
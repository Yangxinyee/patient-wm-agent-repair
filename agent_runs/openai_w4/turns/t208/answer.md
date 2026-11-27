The recovered notes show that head 1 was chosen because the other heads could compensate for its factual output. That makes sharing it plausible, not yet safe. I’ll now inspect only the decoder and gating details needed for construction; the previous diagnostic was truncated.

```python
print("COMPACT NE DECODER LOCATION")
for name, obj in list(globals().items()):
    if isinstance(obj, dict):
        matches = [
            (k, v) for k, v in obj.items()
            if isinstance(k, tuple) and k and k[0] == "norepinephrine"
            and isinstance(v, np.ndarray) and v.shape == (577,)
        ]
        if matches:
            print(name, [(k, float(np.linalg.norm(v))) for k, v in matches])
            globals()["ne_decoder_dict183"] = obj
            globals()["ne_decoder_name183"] = name

print("\nGATE FIT STRUCTURE")
v183 = gate_fits180[100.0]
print(type(v183).__name__)
if isinstance(v183, dict):
    print([(k, getattr(v, "shape", type(v).__name__)) for k, v in v183.items()])
elif isinstance(v183, (tuple, list)):
    print([(i, getattr(v, "shape", type(v).__name__)) for i, v in enumerate(v183)])
else:
    print(getattr(v183, "shape", None))

print("\nCHANNEL FORMULA")
print(dis.Bytecode(norm_latent104).dis())

print("\nCURRENT POSTERIOR WIRING")
w0183 = load("z_post.0.weight")
b0183 = load("z_post.0.bias")
w2183 = load("z_post.2.weight")
wi183 = load("z_in.weight")
for r in (96, 184, 156, 89):
    print("hidden", r, "bias", float(b0183[r]),
          "ctx/e/u norms", [float(np.linalg.norm(w0183[r, j:j+192])) for j in (0,192,384)],
          "latent-axis output", float(latent_axis104 @ w2183[:32, r]))
print("z_in singular values", np.linalg.svd(wi183, compute_uv=False)[:5])
print("z_in axis image norm", np.linalg.norm(wi183 @ latent_axis104))
print("signature/carrier norms, dot", np.linalg.norm(signature104),
      np.linalg.norm(carrier104), signature104 @ carrier104)
print("latent_axis norm", np.linalg.norm(latent_axis104))

print("\nNE TARGET / ACTION OBJECT NAMES")
for name, obj in list(globals().items()):
    if not re.search(r"(?:93|94|95|96|97|175|176|177|178|180)$", name):
        continue
    if isinstance(obj, dict):
        print(name, "dict", list(obj)[:6])
    elif isinstance(obj, (np.ndarray, torch.Tensor)) and (
        any(t in name.lower() for t in ("raw", "target", "gate", "q", "dose", "rate", "coef", "mask"))
        or name.endswith("180")
    ):
        print(name, tuple(obj.shape))

print("\nTRANSITION CAPTURE BYTECODE")
print(dis.Bytecode(capture_transition102).dis()[:2600])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 183\n"
        "Run-182 recovered the compensated head-1 construction, but its "
        "diagnostic output was truncated before the required decoder/gating "
        "details. Repeating those details compactly; no weights changed.\n"
    )
```
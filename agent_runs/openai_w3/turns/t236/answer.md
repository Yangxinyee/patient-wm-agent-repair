The hourly shifts confirm that insulin halving is already wrong-signed at hour 13 and remains weak later. Simple under-scaling cannot explain both insulin readouts. I’ll inspect the carrier and factual-compensation calculations; a compensation term that follows the edited dose could erase plausible reductions while leaving larger increases intact.

```python
import dis
import io

print("=== HELPER IMPLEMENTATIONS ===")
for name, cap in [("latent206", 2400), ("prelude155", 3500)]:
    fn = globals()[name]
    print(name, "globals:", fn.__code__.co_names,
          "defaults:", fn.__defaults__)
    buf = io.StringIO()
    dis.dis(fn, file=buf)
    print(buf.getvalue()[:cap])

print("\n=== DECODER AND EDIT MAGNITUDES ===")
print("rows199:", rows199, "cols199:", cols199)
r211_dec = np.asarray(restricted199)
r211_nz = np.flatnonzero(np.max(np.abs(r211_dec), axis=1) > 1e-7)
print("restricted decoder nonzero rows:", r211_nz)
print("restricted decoder values:", r211_dec[r211_nz])
for name in ["b193", "jo193", "jp193", "dd193", "correction193",
             "carrier154", "repair_pilot_strength"]:
    if name in globals():
        a = np.asarray(globals()[name])
        print(name, a.shape, "norm", np.linalg.norm(a),
              "range", (a.min(), a.max()))
for name in ["core.prelude.0.ffn.2.weight",
             "core.prelude.0.ffn.2.bias",
             "core.prelude.0.attn.out.weight"]:
    print(name, "edit norm",
          float(torch.linalg.vector_norm(
              torch.as_tensor(load(name)) -
              torch.as_tensor(load_original(name)))))

print("\n=== RELEVANT ARRAY/FUNCTION INVENTORY ===")
r211_names = [
    k for k in list(globals()) if not k.startswith("_")
    and k.endswith(("153", "187", "192", "196", "198"))
    and (hasattr(globals()[k], "shape") or callable(globals()[k]))
]
for k in sorted(r211_names):
    v = globals()[k]
    if callable(v):
        print(k, "callable", v.__code__.co_names if hasattr(v, "__code__") else "")
    else:
        print(k, tuple(v.shape))

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n\nPython run 211; forwards 14. Hourly insulin half shift is "
        "-1.796 at hour 13 and remains weak throughout, whereas x2 "
        "grows to -26.18 by hour 24. Investigating asymmetric carrier "
        "or dose-following factual compensation rather than assuming "
        "one scalar gain can repair both. No weights changed.\n"
    )
```
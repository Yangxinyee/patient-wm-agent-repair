The caches contain original hidden activations and logits under 32 latent draws. That permits a gradient-free test: predict the correction’s factual component from those activations, then cancel it while constraining sensitivity to the dose probe. I’ll first verify the correction’s algebra and cache alignment. If cancellation also removes the dose response, this approach is unsuitable.

```python
print("=== COMPLETE FACTUAL CACHES ===")
r237_caches = {}
for rc_name, rc_obj in list(globals().items()):
    if (isinstance(rc_obj, dict)
        and all(k in rc_obj for k in ["X", "H0", "L0", "y", "meta", "loss0"])):
        r237_caches[rc_name] = rc_obj
        rc_meta = rc_obj["meta"]
        rc_sid = next((k for k in ["stay_id", "stay"] if k in rc_meta.columns), None)
        print(rc_name,
              {k: tuple(rc_obj[k].shape) for k in ["X", "H0", "L0", "y"]},
              "meta columns:", list(rc_meta.columns),
              "stay range:", (int(rc_meta[rc_sid].min()),
                              int(rc_meta[rc_sid].max())) if rc_sid else None)

print("\n=== NE PAIR GEOMETRY ===")
r237_W0 = r235_safe["attr_heads.value_bin.0.weight"].numpy()
r237_b0 = r235_safe["attr_heads.value_bin.0.bias"].numpy()
r237_W2 = r235_safe["attr_heads.value_bin.2.weight"].numpy()
r237_common = (r237_W0[48]+r237_W0[171])/2
r237_signal = (r237_W0[48]-r237_W0[171])/2
r237_cb = float((r237_b0[48]+r237_b0[171])/2)
r237_sb = float((r237_b0[48]-r237_b0[171])/2)
r237_tilt = (r237_W2[:,48]-r237_W2[:,171])/2
print("common norm / bias:", np.linalg.norm(r237_common), r237_cb)
print("signal norm / bias:", np.linalg.norm(r237_signal), r237_sb)
print("signal response to raw NE probe:", r237_signal @ r230_d)
print("common response to raw NE probe:", r237_common @ r230_d)
print("output pair sum norm:", np.linalg.norm(r237_W2[:,48]+r237_W2[:,171]))
print("tilt:", np.round(r237_tilt, 4))
print("NE probe norm:", np.linalg.norm(r230_d))
print("gate scale vs r231 gate:",
      (r237_common @ r231_gate_v)/(r231_gate_v @ r231_gate_v))
print("c210_eval argument defaults:",
      [type(v).__name__ for v in (c210_eval.__defaults__ or ())])

print("\n=== OTHER DOSE-PROBE NAMES ===")
print(" ".join(
    n for n in globals()
    if any(k in n.lower() for k in ["probe", "rawd", "raw_d", "d_ins", "ins_d"])
    and not n.startswith("rc_")
)[-1400:])

print("\n=== SELECTED NOTEBOOK HEADINGS ===")
rc_lines_now = Path("/work/notes.md").read_text().splitlines()
rc_chunks = []
for rc_i, rc_s in enumerate(rc_lines_now):
    if rc_s.startswith("##") and any(
        str(j) in rc_s for j in [177, 188, 189, 190, 191, 196, 207, 209, 210]
    ):
        rc_chunks.append("\n".join(rc_lines_now[rc_i:rc_i+4]))
print("\n".join(rc_chunks)[-2500:])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 237: nonlinear factual-cancellation preparation\n"
            "Inspected factual caches and isolated NE pair geometry. Proposed "
            "training-only ridge regression on original hidden features, constrained "
            "to preserve finite-difference dose sensitivity. This is a hypothesis, "
            "not a verified rollout repair. No weights changed.\n")
```
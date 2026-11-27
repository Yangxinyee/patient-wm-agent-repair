The restored joint edit also fails on this sample: NLL rises by about 0.00523. I’ll reduce insulin strength, increase its halving-sensitive component, and moderately strengthen norepinephrine. This trade-off is useful only if paired tests show safer factual fit and better agreement with the record-derived effects.

```python
with np.load("/work/baseline77.npz") as a:
    base268 = a["delta"].copy()
with np.load("/work/insulin_candidate63.npz") as a:
    balanced268 = a["delta"].copy() * (1.6 / float(a["strength"]))
with np.load("/work/ne_candidate151.npz") as a:
    ne268 = a["ne_delta"].copy()

original_b268 = np.asarray(load_original("core.b_in.weight"), dtype=np.float32).copy()

def install_rebalanced268(insulin_scale, insulin_blend, ne_scale, filename):
    idelta = insulin_scale*((1-insulin_blend)*base268 + insulin_blend*balanced268)
    ndelta = ne_scale*ne268
    delta = idelta + ndelta
    save_edit("core.b_in.weight", (original_b268 + delta).astype(np.float32))
    np.savez("/work/"+filename, delta=delta, insulin_delta=idelta, ne_delta=ndelta,
             insulin_scale=insulin_scale, insulin_blend=insulin_blend, ne_scale=ne_scale)
    print("Installed:", insulin_scale, insulin_blend, ne_scale,
          "delta norms:", float(np.linalg.norm(idelta)), float(np.linalg.norm(ndelta)))
    return delta

trial268 = install_rebalanced268(.76, .60, .40, "joint268.npz")
o268 = pd.read_csv("/work/fwd_33/factual_original.csv")
e268 = pd.read_csv("/work/fwd_33/factual_edited.csv")
cost268 = float(e268.nll.mean()-o268.nll.mean())
print("Joint236 factual cost on tuning100..199:", cost268)
print("Current edits:", edited())
print("Max installed-delta error:",
      float(np.max(np.abs(np.asarray(load("core.b_in.weight"))-original_b268-trial268))))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 268: safer balanced joint trial\n"
            f"Forward33 joint236 factual cost={cost268:.8f} on tuning100..199; "
            "this exceeds .005, so joint236 is not robustly safe. "
            "Installed I_scale=.76, I_blend=.60, NE_scale=.40 using existing "
            "record-derived closed-form corrections; no prelude edits. "
            "Saved joint268.npz. Unaccepted pending paired factual and rollout tests.\n"
            "Python runs: 268; forwards: 33.\n")
```
The validated recurrent edit is rank two, with separate glucose and MAP components available. I’ll inspect its response calibration and feature centering next. This will distinguish two possible causes of the MAP failure: a poor dose feature versus a poor downstream readout direction. A useful replacement should respond to half doses as well as doubled doses, with little change on factual inputs.

```python
import dis

def rr_details():
    print("=== EXACT HELPER IMPLEMENTATIONS ===")
    for name in ["r256_predict", "r244_read_pre", "r250_fn",
                 "r245_prediction", "action230", "core228"]:
        fn = globals()[name]
        print("\nFUNCTION", name)
        try:
            text = inspect.getsource(fn)
            print(text[:1800])
        except Exception:
            instructions = list(dis.get_instructions(fn))
            print("globals:", fn.__code__.co_names)
            print("locals:", fn.__code__.co_varnames)
            limit = 48 if name == "core228" else 65
            for ins in instructions[:limit]:
                print(ins.opname, ins.argrepr[:100])

    print("\n=== ROUTING CALIBRATION ===")
    for name in ["r245_R", "r245_need", "r245_response", "r245_Q",
                 "r245_delta", "r244_targets", "r244_weights"]:
        obj = globals().get(name)
        if obj is not None:
            a = obj.detach().cpu().numpy() if isinstance(obj, torch.Tensor) else np.asarray(obj)
            print(name, "shape", a.shape,
                  "values", a if a.size <= 20 else
                  (float(a.min()), float(a.mean()), float(a.max())))
    for name in ["r245_V", "r245_dirs", "r245_G", "r255_CI", "r255_CN"]:
        a = np.asarray(globals()[name])
        print(name, "norm", np.linalg.norm(a),
              "column norms", np.linalg.norm(a, axis=0)[:8])

    print("\n=== FACTUAL FEATURE CENTERING ===")
    for name, v in [("current", r245_V),
                    ("slot_fit", r257_fits[(0.01, 1.0)])]:
        if isinstance(v, dict):
            print(name, "dict keys", list(v.keys()))
            continue
        v = np.asarray(v)
        p = r257_Fm @ v
        print(name, "shape", v.shape, "mean", p.mean(0),
              "sd", p.std(0), "rms", np.sqrt((p*p).mean(0)))

    print("\n=== CASE METADATA ===")
    for key, value in r245_cases.items():
        print(key, {k: tuple(v.shape) for k, v in value.items()
                    if isinstance(v, (np.ndarray, torch.Tensor))})
    print("Y first/last blocks:", r257_Ym[:320].mean(0),
          r257_Ym[-320:].mean(0))
    print("weight range:", np.min(r257_weights), np.max(r257_weights))

    with open("/work/notes.md", "a") as f:
        f.write("\n## Run 260: inspect recurrent calibration and centering\n"
                "- Separate glucose and MAP routes are retained; investigating MAP asymmetry.\n"
                "- Inspecting exact prediction helpers, route norms and factual feature means.\n"
                "- No weights changed. Python runs: 260; forward passes: 21.\n")

rr_details()
```
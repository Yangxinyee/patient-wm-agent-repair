The displayed output repeats the MAP result and does not show a glucose continuation. I’ll check for a cached glucose fit, run it only if needed, and use the correct hourly index name, `t`. A stable glucose estimate would support the current insulin target; further drift would call it into question.

```python
import contextlib
import io
import time

if "latent_ins158" not in globals():
    capture158 = io.StringIO()
    start_time158 = time.perf_counter()
    with contextlib.redirect_stdout(capture158):
        latent_ins158 = fit_latent_record_model(
            panel, ["glucose_finger", "glucose_lab"], "insulin",
            fixed_b=-1.8, iterations=96,
            initial_parameters=latent_fits["insulin"]["parameters"]
        )
    print("Glucose fit seconds:", round(time.perf_counter()-start_time158, 2))
else:
    print("Using cached glucose continuation.")

print("=== GLUCOSE CONVERGENCE ===")
print(latent_ins158["log"].iloc[[0, 23, 47, 71, 95]].to_string(index=False))
print("Final parameters:", latent_ins158["parameters"])

print("\n=== EXACT ROLLOUT FILE FORMAT ===")
rollout158 = pd.read_csv("/work/fwd_17/dose_edit_edited.csv")
print(rollout158.shape, rollout158.columns.tolist())
print(rollout158.head(3).to_string(index=False))
print("Panel hours:", panel.index.get_level_values("t").unique().tolist())

print("\n=== CONSTRAINED-DECODER OBJECTS ===")
count158 = 0
for name158, obj158 in list(globals().items()):
    if name158.endswith(("145", "146", "147", "148", "149", "150")):
        if isinstance(obj158, pd.DataFrame):
            print(name158, obj158.shape, obj158.columns.tolist())
        elif isinstance(obj158, dict):
            print(name158, "dict keys:", list(obj158.keys())[:10])
        elif hasattr(obj158, "shape"):
            print(name158, type(obj158).__name__, tuple(obj158.shape))
        elif callable(obj158):
            print(name158, inspect.signature(obj158))
        else:
            continue
        count158 += 1
        if count158 >= 35:
            break

print("\n=== FEATURE ARRAYS ===")
count158 = 0
for name158, obj158 in list(globals().items()):
    if (name158.startswith("E") or "probe" in name158.lower()) and hasattr(obj158, "shape"):
        print(name158, type(obj158).__name__, tuple(obj158.shape))
        count158 += 1
        if count158 >= 25:
            break

marker158 = "## Runs 157–158: extended record-only persistence fits"
if marker158 not in Path("/work/notes.md").read_text():
    with open("/work/notes.md", "a") as f158:
        f158.write(
            f"\n{marker158}\n"
            f"NE fixed b=3.6: rho {latent_ne157['parameters']['rho']:.6f} "
            "after 48 additional iterations; near numerical convergence. "
            f"Insulin fixed b=-1.8: rho {latent_ins158['parameters']['rho']:.6f} "
            "after 96 additional iterations. These are assumption-dependent "
            "record-only estimates. Run157's error was only a reporting lookup "
            "of 'hour' instead of 't'. World-model weights unchanged.\n"
        )
```

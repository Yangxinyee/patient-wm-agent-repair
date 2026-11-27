The persistence estimates disagree somewhat: the latent fit gives about 0.96 for glucose and 0.84 for MAP, while the sensor-based check suggests higher values. I’ll first continue the MAP fit and inspect its convergence. Little further movement would support its current target; substantial drift would show that the earlier estimate was premature.

```python
import contextlib
import io
import time

stored_parameters157 = {}
def collect_parameters157(obj, label, depth=0):
    if not isinstance(obj, dict):
        return
    if "rho" in obj and "b" in obj:
        stored_parameters157[label] = obj
        return
    if depth < 3:
        for key, value in obj.items():
            if isinstance(value, dict):
                collect_parameters157(value, f"{label}.{key}", depth+1)

for name157, obj157 in list(globals().items()):
    if name157 != "stored_parameters157":
        collect_parameters157(obj157, name157)

print("=== STORED PARAMETERS, COMPACT ===")
for name157, par157 in stored_parameters157.items():
    print(name157, "b:", par157["b"], "rho:", par157["rho"],
          "Q:", par157.get("Q"), "R:", par157.get("R"))

assert isinstance(fit, dict) and "parameters" in fit
assert fit["parameters"]["b"] > 0
ne_start157 = {
    key: value.copy() if isinstance(value, np.ndarray) else value
    for key, value in fit["parameters"].items()
}
capture157 = io.StringIO()
start_time157 = time.perf_counter()
with contextlib.redirect_stdout(capture157):
    latent_ne157 = fit_latent_record_model(
        panel, ["map_arterial", "map_cuff"], "norepinephrine",
        fixed_b=3.6, iterations=48, initial_parameters=ne_start157
    )
print("\n=== NOREPINEPHRINE PERSISTENCE CONTINUATION ===")
print("Seconds:", round(time.perf_counter()-start_time157, 2))
print(latent_ne157["log"].tail(6).to_string(index=False))
print("Final parameters:", latent_ne157["parameters"])

print("\n=== CURRENT DECODER STRUCTURE ===")
entry157 = constraint_candidates147[(0.3, 0.0001)]
print("Entry type:", type(entry157).__name__)
if isinstance(entry157, dict):
    for key157, value157 in entry157.items():
        print(key157, type(value157).__name__, getattr(value157, "shape", None),
              "" if hasattr(value157, "shape") else str(value157)[:250])
elif isinstance(entry157, (tuple, list)):
    for i157, value157 in enumerate(entry157):
        print(i157, type(value157).__name__, getattr(value157, "shape", None),
              "" if hasattr(value157, "shape") else str(value157)[:250])
else:
    print("Shape:", getattr(entry157, "shape", None))

print("\n=== PANEL ALIGNMENT ===")
print(panel.head(3).to_string())
print("Index names:", panel.index.names)
print("Hours:", panel.index.get_level_values("hour").unique().tolist())

with open("/work/notes.md", "a") as f157:
    f157.write(
        "\n## Run 157: record-only MAP persistence convergence\n"
        "Corrected pre-dose DML estimates support NE +3.5 to +3.9/rate; "
        "insulin approximately -1.6 to -2.0/rate, with insulin RD lab check "
        "and observation-policy falsification. Persistence remains assumption-dependent. "
        f"Extended NE Gaussian latent fit by 48 iterations at fixed b=3.6: "
        f"rho {latent_ne157['parameters']['rho']:.6f}. No world-model weights changed.\n"
    )
```

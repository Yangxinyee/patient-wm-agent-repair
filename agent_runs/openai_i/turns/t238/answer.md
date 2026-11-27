The recovered record analyses support an insulin effect near −2 per rate unit and a norepinephrine effect near +3 after startup, with identification caveats. The output was truncated before the implementation details, so I’ll retrieve those compactly. I need to verify that each private coordinate tracks dose innovation before assigning it a response.

```python
import inspect
import re
from pathlib import Path

print("RELEVANT RETAINED NAMES")
for suffix212 in [194, 196, 198, 201, 202, 208, 209]:
    names212 = sorted(
        n for n in list(globals())
        if re.search(rf"(?<!\d){suffix212}$", n)
    )
    print(suffix212, " ".join(names212))

print("\nHELPER CONTRACTS")
for name212 in ["mlp136", "call_core161", "batch_q182", "batch_replay182"]:
    fn212 = globals().get(name212)
    if callable(fn212):
        print(name212, str(inspect.signature(fn212)))
        print("local names:", fn212.__code__.co_varnames)
        print("global names:", fn212.__code__.co_names)
        print("constants:", str(fn212.__code__.co_consts)[:600])

print("\nRECENT FUNCTION CONTRACTS")
for name212 in sorted(list(globals())):
    m212 = re.search(r"(\d+)$", name212)
    fn212 = globals()[name212]
    if m212 and 194 <= int(m212.group(1)) <= 209 and callable(fn212):
        print(name212, str(inspect.signature(fn212)),
              "globals:", fn212.__code__.co_names)

print("\nSELECTED ARRAY CONTRACTS")
for name212 in ["input196", "all_z201", "zin_bias_base201",
                "neutral209", "post209", "private209",
                "innov196", "innov201", "innov209",
                "base_tokens209", "tokens209"]:
    if name212 in globals():
        obj212 = globals()[name212]
        print(name212, type(obj212).__name__,
              getattr(obj212, "shape", None))
        if isinstance(obj212, dict):
            print("keys:", list(obj212)[:12])

print("\nCURRENT EDIT GEOMETRY")
for name212 in edited():
    a212 = np.asarray(load(name212))
    d212 = a212 - np.asarray(load_original(name212))
    print(name212, a212.shape,
          "delta RMS", round(float(np.sqrt(np.mean(d212*d212))), 6))
zin212 = np.asarray(load("z_in.weight"))
print("innovation column norms", np.linalg.norm(zin212[:, [16, 12]], axis=0))

print("\nEVIDENCE ABOUT DYNAMICS AND ROLLOUT TARGETS")
parts212 = re.split(r"(?m)(?=^## )", Path("/work/notes.md").read_text())
selected212 = [
    s for s in parts212
    if re.search(r"persistence|impulse|steady|rollout target|carryover|carry-over|dynamic", s, re.I)
]
for s212 in selected212[:4] + selected212[-3:]:
    print(s212[:800])

with open("/work/notes.md", "a") as f212:
    f212.write(
        "\n## Python 212 — compact recovery after truncated output\n"
        "- Earlier record estimates support approximately -2 insulin and +3 NE per rate unit, not medical priors.\n"
        "- Retrieving private-coordinate and replay contracts before numerical response calibration.\n"
        "- No additional weight changes.\n"
    )
```
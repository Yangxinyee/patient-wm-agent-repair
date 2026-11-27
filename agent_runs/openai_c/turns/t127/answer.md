The records and architecture point to a persistent drug effect, but the tested head edits have not restored it. I’ll review earlier pathway tests and how the saved directions were estimated before selecting another intervention. A dose direction that survives adjustment for prior state is a better candidate than an unadjusted drug association.

```python
from pathlib import Path
import inspect, numpy as np

text90 = Path("/work/notes.md").read_text()
sections90 = text90.split("\n## ")
print("EARLIER EXPERIMENT SUMMARIES")
for section in sections90:
    if any(x in section.lower() for x in
           ["gain 8", "gain8", "numeric", "query", "diagonal", "c_out",
            "b_in", "action", "transition", "persistence", "gain 4"]):
        lines90 = section.splitlines()
        print("## " + lines90[0])
        for line90 in lines90[1:]:
            if any(x in line90.lower() for x in
                   ["nll", "rollout", "rejected", "scale", "gain", "installed",
                    "persistence", "effect", "insulin", "norepinephrine"]):
                print(line90[:420])
        if section.startswith("Python run 59"):
            break

print("\nREMAINING HEAD / ACTION TENSORS")
for n in NAMES:
    if any(s in n for s in ["last_numeric", "value_bin", "query_norm"]):
        print(n, SHAPES[n])

print("\nSTATE-DIRECTION HELPERS")
for fn90 in ["conditional_transition_slopes72", "apply_lowrank_left_transform",
             "apply_lowrank_right_transform", "install_latent_gain"]:
    fun90 = globals().get(fn90)
    if fun90 is not None:
        print(fn90, inspect.signature(fun90))
        print("names:", fun90.__code__.co_names)
        print("constants:", [x for x in fun90.__code__.co_consts
                              if isinstance(x, (str, int, float, tuple))][:14])

print("\nCOMPACT SAVED MODELS")
for k90 in ["repair_signal", "state_geom", "empirical_head59",
            "sensor_designs72", "dose_coeff_compare71"]:
    ob90 = globals().get(k90)
    if isinstance(ob90, dict):
        print(k90)
        for a90, val90 in ob90.items():
            if isinstance(val90, dict):
                print(a90, {b90: (v90.shape if isinstance(v90, np.ndarray)
                                  else str(v90)[:100])
                             for b90, v90 in val90.items()})
            else:
                print(a90, val90.shape if isinstance(val90, np.ndarray)
                      else str(val90)[:200])

with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 90: recurrent-edit design review\n"
            "- Numeric gain 4 remains the safe baseline; no operations added.\n"
            "- Reviewing earlier gain/query/diagonal experiments and exact direction-helper signatures.\n"
            "- Forward passes used: 34; 16 remain.\n")
```
The paired rollout supports this route: insulin shifts moved to −9.02/+4.49 and norepinephrine shifts to +3.37/−1.56, all in the record-supported directions. They remain smaller than the earlier accumulation estimates. Before increasing gains, I’ll revisit those estimates and the decoder diagnostics; robust persistence evidence and a manageable factual cost would justify a stronger edit.

```python
import re
notes138 = Path("/work/notes.md").read_text()
sections138 = re.split(r"(?m)(?=^## )", notes138)
wanted138 = ["Run 5:", "Run 6:", "Run 11:", "Run 12:", "Run 22:",
             "Run 33:", "Run 39:", "Runs 128"]
budget138 = 6500
for marker138 in wanted138:
    sec138 = next((s for s in sections138 if s.startswith("## " + marker138)), "")
    if sec138 and budget138 > 0:
        shown138 = sec138[:min(1300, budget138)]
        print(shown138)
        budget138 -= len(shown138)
print("=== CACHED ESTIMATES / FIT ===")
for name138 in ["sensor_iv_persistence", "iv_persistence_table",
                "linear_rollout_targets", "record_targets", "record_impulse_targets",
                "fit128"]:
    if name138 in globals():
        obj138 = globals()[name138]
        if isinstance(obj138, pd.DataFrame):
            print(name138, obj138.head(12).to_string(index=False)[:1000])
        elif isinstance(obj138, dict):
            print(name138, {k: (np.shape(v) if hasattr(v, "shape") and np.size(v)>20
                               else repr(v)[:900]) for k, v in obj138.items()})
        else:
            print(name138, repr(obj138)[:700])
print("=== FIT / STEERING VARIABLES ===")
print([n for n in sorted(globals()) if any(s in n.lower() for s in
      ["fit1", "feature1", "prelude", "b_in", "gain", "sensitiv", "steer", "install"])][-100:])
with open("/work/notes.md", "a") as f138:
    f138.write("\n## Forward 14: corrected recurrent steering succeeds directionally\n"
               "99 tuning stays, paired exact rollouts: original insulin x2/half "
               "-3.226/+1.825, NE +1.373/-0.846; corrected edited gains [-5,+8] "
               "give -9.016/+4.485 and +3.372/-1.557. Factual cost from forward13 "
               "+0.00166. Retain this as a viable fallback; audit record-supported "
               "accumulation targets before stronger gains.\n")
```

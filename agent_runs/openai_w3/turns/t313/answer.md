The halved-path candidate passes the factual limit (+0.00128 nats), but offers no rollout improvement. I’ll restore the preceding validated candidate and inspect how its dose carriers enter the state. If local responses are already symmetric, the next correction must address recurrence rather than dose decoding.

```python
import io, dis
for n, v in checkpoint273.items():
    save_edit(n, v)
print("Restored checkpoint273; edited tensors:", len(edited()))

print("\nIMPLEMENTATION")
for name in ["latent206", "hidden220", "parts254", "state_update82", "readout134"]:
    f = globals()[name]
    print("\nFUNCTION", name, inspect.signature(f))
    try:
        text = inspect.getsource(f)
    except Exception:
        stream = io.StringIO()
        dis.dis(f, file=stream)
        text = stream.getvalue()
    print(text[:1500])

print("\nRETAINED SOURCE253")
print(source253[:1800])

print("\nEDIT NORMS")
for n in edited():
    a, b = as_numpy(load(n)), as_numpy(load_original(n))
    print(n, "original", round(float(np.linalg.norm(b)), 2),
          "delta", round(float(np.linalg.norm(a-b)), 2))

print("\nNOTEBOOK RUNS 210-225")
for line in lines275:
    if re.search(r"Python runs?: (21\d|22[0-5])\b|Python run (21\d|22[0-5])\b", line):
        print(line[:250])

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 276; forwards 24. Whole-output halving passed factual (+.001278) but provided no causal improvement; restored checkpoint273 (validated total-response gamma=.1). Recovering carrier and exact recurrence/readout implementation. No new candidate installed.\n")
```
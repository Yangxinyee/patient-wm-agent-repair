This repeats run 196, so the baseline-drift and transport tests are still unreported. The conclusion remains that insulin decoding is accurate but its rollout response is asymmetric. I’ll run those diagnostics next; symmetric local responses combined with baseline drift would implicate recurrent feedback.

```python
import json, pandas as pd, inspect, dis

ro197 = pd.read_csv("/work/fwd_12/dose_edit_original.csv")
re197 = pd.read_csv("/work/fwd_12/dose_edit_edited.csv")
merged197 = re197.merge(ro197, on=["stay_id", "hour", "edit"],
                       suffixes=("_edited", "_original"))
print("=== BASELINE ROLLOUT DRIFT ===")
for label in ["insulin_x2", "insulin_half", "ne_x2", "ne_half"]:
    q = merged197[(merged197.edit == label) & merged197.reached_edited]
    print(label, "baseline original/edited, intervention:",
          np.round([q.factual_original.mean(), q.factual_edited.mean(),
                    (q.factual_edited + q.shift_edited).mean()], 3))
    print("edited baseline quantiles:",
          np.round(q.factual_edited.quantile([0, .1, .5, .9, 1]).to_numpy(), 2))

print("\n=== SYMMETRIC LOCAL TRANSPORT RESPONSES ===")
inj197, labels197 = [], []
for d in [0, 1]:
    for gain in [-2, -1, -.5, -.25, .25, .5, 1, 2]:
        inj197.append(directions187[:, d] * gain)
        labels197.append((d, gain))
resp197 = np.asarray(response186(np.asarray(inj197)))
for label, row in zip(labels197, resp197 - np.asarray(base186)):
    print(label, np.round(row, 4))

print("\n=== ACTION HELPER ===")
for name in ["action_vectors87", "posterior153"]:
    fn197 = globals()[name]
    print(name, inspect.signature(fn197), fn197.__code__.co_names,
          [v for v in fn197.__code__.co_consts
           if isinstance(v, (str, float, int, tuple))])

print("\n=== SCALED-ACTION DICTIONARIES ===")
for name, v in list(globals().items()):
    if isinstance(v, dict) and v and any(
        isinstance(k, tuple) and len(k) == 2 and k[0] in [0, 1]
        for k in list(v)[:5]
    ):
        first = next(iter(v.values()))
        print(name, list(v)[:8],
              tuple(first.shape) if hasattr(first, "shape") else type(first).__name__)

print("\n=== PRELUDE INJECTION AND RETURN ===")
bc197 = list(dis.Bytecode(prelude155))
for pos, ins in enumerate(bc197):
    if ins.opname == "LOAD_FAST" and ins.argval == "injection":
        for x in bc197[max(0, pos-5):pos+16]:
            print(x.opname, x.argrepr)
print("return:")
for x in bc197[-25:]:
    print(x.opname, x.argrepr)

with open("/work/notes.md", "a") as f:
    f.write("\nDiagnostic after run 196; forwards 12. Previous output repeated run 196. Testing baseline drift and symmetric local transport; no weights changed.\n")
```
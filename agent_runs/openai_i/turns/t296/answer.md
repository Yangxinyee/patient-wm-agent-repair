The neutral edit matched the original factual NLL, but both attention-response tests failed by about 0.107 nats. Sharpening the selector apparently made no difference, which conflicts with the local leakage tests. I’ll reinstall that saved candidate explicitly and inspect its GPU states. If it changes states before treatment, the transport is not truly gated and must be rejected or repaired.

```python
sharp262 = np.load("/work/sharp243.npz")
print("SAVED SHARP TENSORS", sharp262.files)
for name262 in list(edited()):
    drop_edit(name262)
for name262 in sharp262.files:
    if name262 in NAMES:
        save_edit(name262, sharp262[name262].astype(np.float32))
print("INSTALLED", edited())
for name262 in ("core.prelude.0.attn.q.weight",
                "core.prelude.0.attn.kv.weight",
                "core.prelude.0.attn.out.weight"):
    if name262 in sharp262.files:
        aa262 = np.asarray(load(name262))
        bb262 = np.asarray(load_original(name262))
        print(name262, "edit_norm", float(np.linalg.norm(aa262-bb262)),
              "saved_difference", float(np.max(np.abs(aa262-sharp262[name262]))))
ids262 = pd.read_csv("/work/fwd_23/factual_original.csv").stay_id.unique()
print("STATE_TEST_STAYS", [int(x) for x in ids262[:12]])
print("CAUSAL_TARGETS", {k:np.asarray(v).tolist() for k,v in causal_targets215.items()})
print("J215", np.shape(J215), "indices215", np.shape(indices215))
print("DIRECTION_NORMS", {k:float(np.linalg.norm(v)) for k,v in mean_directions216.items()})
with open("/work/notes.md", "a") as f:
    f.write("\n## Python 262 — instrument rejected attention candidate\n"
            "- Explicitly installed saved sharp243 only to diagnose the prior factual failure.\n"
            "- It is NOT accepted: previous dNLL was +.10705.\n"
            "- Next GPU state comparison will test pre-treatment transport leakage; neutral209 remains the safe restore point.\n")
```

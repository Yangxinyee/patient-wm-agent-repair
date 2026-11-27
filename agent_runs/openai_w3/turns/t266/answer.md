The kernel retains the checkpoints and paired-dose features needed for controlled tests. I’ll inspect the replay helpers and compare the actual transport with the installed directions. If a scale or mixing error explains the mismatch, correcting it should improve both dose directions without sacrificing factual fit.

```python
import dis, io
for name238 in ["raw_hidden228", "latent206", "hidden220",
                "response_raw234", "predict_E219"]:
    f238 = globals()[name238]
    print("\nHELPER", name238, inspect.signature(f238))
    try:
        text238 = inspect.getsource(f238)
    except Exception:
        text238 = dis.Bytecode(f238).dis()
    print(text238[:1050])

print("\nTRANSPORT COMPARISONS")
for name238 in ["dd193", "correction193", "dirs_full228", "r235_transport"]:
    v238 = np.asarray(globals()[name238])
    print(name238, "norms", np.round(np.linalg.norm(v238, axis=0), 3))
    print("map to actual transport",
          np.round(np.linalg.lstsq(v238, np.asarray(r235_transport), rcond=None)[0], 4))
print("\nEDIT DIFFERENCE NORMS")
for name238 in edited():
    v238 = as_numpy(load(name238))
    d238 = v238 - as_numpy(load_original(name238))
    print(name238, tuple(v238.shape), "change", round(float(np.linalg.norm(d238)), 4))

print("\nFFN CHANGED COLUMN NORMS")
dff238 = as_numpy(load("core.prelude.0.ffn.2.weight")) - as_numpy(
    load_original("core.prelude.0.ffn.2.weight"))
ci238 = np.argsort(np.linalg.norm(dff238, axis=0))[-12:]
print(list(zip(ci238.tolist(), np.round(np.linalg.norm(dff238[:,ci238], axis=0), 3))))
print("\nRELEVANT NAMES")
print(" ".join(sorted(k for k in dict(globals())
                     if any(s in k.lower() for s in
                            ["carrier", "gain", "compens", "direction",
                             "payload", "neuron", "decode", "insert"]))))
with open("/work/notes.md", "a") as f:
    f.write("\nPython run 238; forwards 16. Inspecting replay helpers, installed direction scales, and FFN edit structure to diagnose bidirectional rollout asymmetry. No weights changed.\n")
```
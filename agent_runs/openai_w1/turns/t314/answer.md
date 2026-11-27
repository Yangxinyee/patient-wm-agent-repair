The recovered record estimates consistently give negative insulin effects and positive norepinephrine effects, with responses spread over several hours. Those remain observational estimates, not known simulator truth. I’ll now trace the exact prelude computation and locate its cached inputs. If the mismatch comes from incorrect slot construction, rebuilding those features should sharply reduce unintended factual injections.

```python
import dis, numpy as np, torch

print("=== LOCAL REPLAY HELPERS ===")
for n in sorted(list(globals())):
    if n.startswith("rx_"):
        o = globals()[n]
        if callable(o) and hasattr(o, "__code__"):
            print(n, "fn", o.__code__.co_varnames[:o.__code__.co_argcount],
                  "uses", o.__code__.co_names)
        elif hasattr(o, "shape"):
            print(n, tuple(o.shape))
        elif isinstance(o, dict):
            print(n, "dict", list(o.keys())[:5])

print("\n=== EXACT HELPER OPERATIONS ===")
for n in ["prelude269", "coda269", "posterior_token270",
          "r244_attention_features", "rx_attn", "rx_layer2", "rx_norm"]:
    if n in globals() and callable(globals()[n]):
        fn = globals()[n]
        print(n, "CONSTANTS", fn.__code__.co_consts)
        print(" | ".join(i.opname + " " + i.argrepr for i in dis.get_instructions(fn)))

print("\n=== TOKEN / STATE CACHES ===")
for n in sorted(list(globals())):
    o = globals()[n]
    if hasattr(o, "shape") and tuple(o.shape) in [
        (5215, 7, 192), (5215, 4, 192), (48, 7, 192),
        (48,), (48, 576), (5215, 576)]:
        if isinstance(o, (np.ndarray, torch.Tensor)):
            a = o.detach().cpu().numpy() if isinstance(o, torch.Tensor) else o
            print(n, tuple(a.shape), "RMS", float(np.sqrt(np.mean(a.astype(float)**2))),
                  "first", a.reshape(-1)[:3])

print("\n=== SLOTS AND STATE ALIGNMENT ===")
for n in ["core.slots", "core.flow.a", "core.delta_in"]:
    a = load_original(n).numpy()
    print(n, a.shape, "RMS", float(np.sqrt(np.mean(a*a))), "first", a.reshape(-1)[:4])
print("rr_state4 keys", rr_state4.keys())
print("rr_keys4 first", rr_keys4[:4])
print("rr_state4 first IDs/hours", rr_state4["stay_id"][:4], rr_state4["hour"][:4])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 275\nRecovered six-lag observational kernels: insulin [-1.850,-2.256,-1.943,-1.561,-1.022,-.690], NE [3.351,2.621,1.717,.943,.494,.409]. Tracing exact replay and token caches to repair the input mismatch; no new weights changed. Counts: 275 Python runs, 29 forwards.\n")
```
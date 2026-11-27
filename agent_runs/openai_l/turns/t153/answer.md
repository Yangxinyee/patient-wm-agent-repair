The record-derived responses are delayed for insulin and shorter-lived for norepinephrine, so a one-step correction would be inadequate. The attention reconstruction preserved factual NLL. I’ll now check how the four-state controller connects to the measured state responses; matching those responses would justify testing a causal injection.

```python
print("=== CONTROLLER ARRAY SHAPES ===")
names141 = [
    "quiet_basis125", "constraints125", "null125", "sub125", "ev125",
    "Jpre125", "Fflow125", "Xstate125", "Ystate125", "Sstate125",
    "rhs_state125", "controllers125", "Bdose125", "gn125", "bn125",
    "constant125", "constvec125", "raw125", "pooled125", "newpool125",
    "Jselected118", "Vsignal118", "transmit122", "Cpre118", "Cstate118",
    "Chead118", "CinvJ118", "state_rows118", "hrows118",
    "ids118", "ids103", "kernels19", "ar2_predictions19"
]
for n in names141:
    v = globals().get(n)
    if isinstance(v, (np.ndarray, torch.Tensor)):
        a = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
        print(n, tuple(a.shape), "norm", round(float(np.linalg.norm(a)), 5))
        if a.size <= 40:
            print(np.round(a, 5))
    elif isinstance(v, dict):
        print(n, "dict", {str(k): (tuple(z.shape) if hasattr(z, "shape") else str(z)[:180])
                         for k, z in list(v.items())[:12]})
    else:
        print(n, type(v).__name__, str(v)[:300])
print("=== FUNCTION DETAILS ===")
import dis
for n in ["prepare103", "func115", "obj115", "compact_recovery135"]:
    g = globals().get(n)
    if callable(g) and hasattr(g, "__code__"):
        print(n, "names", g.__code__.co_names, "constants", str(g.__code__.co_consts)[:700])
print("=== TRANSITION BYTECODE ===")
describe_code130(transition64, 2200)
print("=== PREPARATION BYTECODE ===")
describe_code130(prepare103, 1800)
print("=== CORE TENSOR SHAPES ===")
for n in NAMES:
    if n.startswith("core.") and any(s in n for s in
                                    ["delta", "a_in", "b_in", "c_out", "prelude_norm", "out_norm"]):
        print(n, SHAPES[n])
print("=== CPU FUNCTIONS CURRENTLY BOUND ===")
for n in ["linear63", "norm63", "attention63", "decode67"]:
    g = globals()[n]
    print(n, g.__name__, g.__code__.co_names)
```

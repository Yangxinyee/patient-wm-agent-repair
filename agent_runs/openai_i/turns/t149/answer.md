The architecture confirms that a posterior edit can affect future readings through the shared recurrent state, and the retained record estimates show multi-hour effects. The output was truncated before the reconstruction helpers appeared. I’ll recover those details next; agreement with the exposed GPU states will determine whether a CPU response calculation is trustworthy.

```python
import inspect
import dis

print("=== CONFIGURATION ===")
print(VOCAB["config"])
print("gap-related weights:",
      [(n, SHAPES[n]) for n in NAMES if "gap" in n.lower()])

print("\n=== CONTROLLED IMPULSE ESTIMATES ===")
for s134 in sections133:
    if s134.startswith("## Lagged-effect investigation"):
        print(s134[:2400])
        break

print("\n=== CORE / RESPONSE FUNCTIONS ===")
for n134, o134 in list(globals().items()):
    if callable(o134) and any(w in n134.lower() for w in
                             ["core", "attention", "block", "step", "repair"]):
        c134 = getattr(o134, "__code__", None)
        if c134 is not None:
            print(n134, c134.co_varnames[:c134.co_argcount], c134.co_names[:20])

print("\n=== SAVED CANDIDATES ===")
for n134 in ["candidate_small131", "candidate_large132"]:
    o134 = globals().get(n134)
    print(n134, type(o134).__name__)
    if isinstance(o134, dict):
        print([(k, getattr(v, "shape", type(v).__name__)) for k, v in o134.items()])

print("\n=== RETAINED EDIT-RELATED OBJECTS ===")
for n134, o134 in list(globals().items()):
    if re.search(r"(12[5-9]|13[01])$", n134):
        if isinstance(o134, (np.ndarray, torch.Tensor, pd.DataFrame)):
            print(n134, type(o134).__name__, tuple(o134.shape))
        elif isinstance(o134, dict):
            print(n134, "dict", list(o134)[:10])
        elif callable(o134):
            c134 = getattr(o134, "__code__", None)
            print(n134, "function",
                  None if c134 is None else c134.co_varnames[:c134.co_argcount])
        elif isinstance(o134, (str, int, float, list, tuple)):
            text134 = repr(o134)
            print(n134, text134[:180])

print("\n=== BUNDLE AND HEAD ALIGNMENT ===")
print("first bundle:", repr(bundle_rows65[0])[:900])
for n134 in ["state_x", "state_hours", "previous_state"]:
    o134 = globals().get(n134)
    print(n134, type(o134).__name__, getattr(o134, "shape", None))
print("reading tables:",
      [(n, tuple(o.shape)) for n, o in list(globals().items())
       if isinstance(o, pd.DataFrame)
       and {"event", "hour", "previous_state"}.issubset(o.columns)][:12])

print("\n=== SMALL HELPER IMPLEMENTATIONS ===")
for n134 in ["repair_linear", "repair_norm", "repair_mlp_act"]:
    o134 = globals().get(n134)
    if callable(o134):
        print("\n", n134)
        dis.dis(o134)
```
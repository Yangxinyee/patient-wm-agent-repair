The balanced head predicts −10.01, +5.09, +2.21 and −0.93. It strengthens norepinephrine doubling slightly, but weakens insulin and leaves norepinephrine halving substantially under-responsive. I’ll inspect the retained fitting objects for a records-fitted calibration that could recover factual margin without suppressing dose responses. Independent factual and rollout tests would decide whether it helps.

```python
import types

with open("/work/notes.md", "a") as f278:
    f278.write("\n## Forward 25–26 / balanced broadened head\n"
               "arch263 + grid269(.25,.25): paired factual on tuning100–199 "
               "1.62067 vs 1.61582, delta +.00485 (4664 readings). "
               "Paired rollout [-10.008,+5.090,+2.209,-.928], original "
               "[-4.072,+2.273,+.615,-.858]. Relative to .25/.15, NE x2 "
               "improves modestly, insulin weakens, and factual margin narrows. "
               "No full causal repair claimed; backup273 remains available.\n")

print("=== FITTING FUNCTIONS ===")
funcnames278 = [
    n for n, v in list(globals().items())
    if isinstance(v, types.FunctionType)
    and re.search(r"fit|head|arch|stat|calib|target|recon|score|nll|eval", n, re.I)
]
print(" ".join(sorted(funcnames278)))

print("=== RELEVANT LIVE ARRAYS / CONTAINERS ===")
names278 = [
    n for n in sorted(list(globals()))
    if re.search(r"263|269|234|179|cache|bins|code|truth|lastbin|target|fitmask|tune", n, re.I)
]
for n278 in names278:
    o278 = globals()[n278]
    if isinstance(o278, np.ndarray):
        print(n278, "array", o278.shape, str(o278.dtype))
    elif isinstance(o278, pd.DataFrame):
        print(n278, "frame", o278.shape, list(o278.columns)[:10])
    elif isinstance(o278, dict):
        print(n278, "dict", repr(list(o278)[:10]))
    elif isinstance(o278, types.FunctionType):
        print(n278, "function", o278.__code__.co_varnames[:o278.__code__.co_argcount])

print("=== GRID269 ENTRY STRUCTURE ===")
entry278 = grid269[(.25, .25)]
print(type(entry278).__name__)
if isinstance(entry278, dict):
    for k278, v278 in entry278.items():
        print(k278, getattr(v278, "shape", None), type(v278).__name__)
elif isinstance(entry278, (tuple, list)):
    for j278, v278 in enumerate(entry278):
        print(j278, getattr(v278, "shape", None), type(v278).__name__,
              repr(v278)[:180] if not hasattr(v278, "shape") else "")

print("=== QUIET-WIRE AND HEAD CONSTRUCTION NOTES ===")
for s278 in sections277:
    heading278 = s278.splitlines()[0]
    if re.search(r"Run[s]? (14[5-9]|15[0-9]|16[0-9]|17[0-2])", heading278):
        print(s278[:320])

print("=== RECENT FACTUAL CSV SCHEMA ===")
for p278 in ("/work/fwd_26/factual_original.csv",
             "/work/fwd_26/factual_edited.csv"):
    d278 = pd.read_csv(p278)
    print(os.path.basename(p278), list(d278.columns), d278.head(1).to_dict("records"))
```
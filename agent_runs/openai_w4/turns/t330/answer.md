The candidate passes factual validation on these stays, but narrowly: +0.00453 nats leaves little margin below 0.005. I’ll recover the longitudinal effect targets and inspect the norepinephrine gate before changing anything else. A gate that suppresses dose innovations during rollout would justify a targeted repair; otherwise I’ll retain the validated fallback.

```python
import dis

print("LONGITUDINAL IDENTIFICATION NOTES")
for number in (136, 137, 138, 139, 140, 141):
    candidates = [s for s in sections281 if re.match(rf"## Run {number}\b", s)]
    for s in candidates:
        print(s[:850])

print("\nNUMERICAL TARGET LINES")
lines283 = notes281.splitlines()
hits283 = [
    i for i, line in enumerate(lines283)
    if re.search(r"target|convolution|bootstrap|direct.kernel|impulse", line, re.I)
    and re.search(r"\d", line)
]
for i in hits283[-14:]:
    print("\n".join(lines283[max(0, i-1):min(len(lines283), i+2)])[:240])

print("\nNE CANDIDATE CONSTRUCTION")
fn283 = symmetric_ne272
print("names", fn283.__code__.co_names)
print("constants", [c for c in fn283.__code__.co_consts if isinstance(c, (str, int, float, tuple))])
print(" | ".join(f"{i.opname} {i.argrepr}" for i in dis.get_instructions(fn283))[:3600])

print("\nINSULIN CANDIDATE GLOBALS")
print(asymmetric_ins279.__code__.co_names)
for k in set(fn283.__code__.co_names + asymmetric_ins279.__code__.co_names):
    if k in globals():
        obj = globals()[k]
        if hasattr(obj, "shape"):
            print(k, type(obj).__name__, tuple(obj.shape))
        elif isinstance(obj, (int, float)):
            print(k, obj)
        elif isinstance(obj, dict):
            print(k, "dict", len(obj), list(obj)[:8])

print("\nRETAINED TARGET / GATE OBJECTS")
for k in list(globals()):
    if re.search(r"(kernel|target|bootstrap|readout|gate.*model|model.*gate|coefne|coef_ne|necoef)", k, re.I):
        obj = globals()[k]
        if hasattr(obj, "shape"):
            print(k, type(obj).__name__, tuple(obj.shape))
        elif isinstance(obj, dict):
            print(k, "dict", len(obj), list(obj)[:6])

with open("/work/notes.md", "a") as f:
    f.write("\n## GPU passes 32–33; Run 283\nAsymmetric insulin 2.25/.75 gate 26 predicts reached-row shifts -11.957/+9.592; NE +.620/-3.188. GPU factual delta +.00453 on 98 tuning stays: passes narrowly. Halving did not decrease as hypothesized. Recovering longitudinal targets and NE gating before further edits; no weights changed.\n")
```

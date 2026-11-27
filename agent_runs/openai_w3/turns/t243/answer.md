The recovered replay computation lets us test drug-direction impulses while holding the surrounding history fixed. I’ll compare equal positive and negative impulses, avoiding rollout feedback. Strong asymmetry here would support a nonlinear transport problem; near symmetry would instead implicate evolving histories.

```python
print("=== CALIBRATION TARGETS ===")
for name217 in ["codes171", "T171", "base186"]:
    obj217 = globals().get(name217)
    if obj217 is not None:
        arr217 = obj217.detach().cpu().numpy() if torch.is_tensor(obj217) else np.asarray(obj217)
        print(name217, arr217.tolist())
print("direction norms", np.linalg.norm(directions186, axis=0).tolist())
print("installed-direction norms", np.linalg.norm(r214_dirs, axis=0).tolist())
print("J186 @ installed directions", (J186 @ r214_dirs).tolist())

scales217 = [-3., -2., -1., -.5, -.25, 0., .25, .5, 1., 2., 3.]
impulses217 = np.stack([
    scale217 * r214_dirs[:, drug217]
    for drug217 in range(2) for scale217 in scales217
]).astype(np.float32)
responses217 = response186(impulses217)
print("\n=== FIXED-HISTORY IMPULSE RESPONSES ===")
for drug217 in range(2):
    block217 = responses217[drug217 * len(scales217):(drug217 + 1) * len(scales217)]
    baseline217 = block217[scales217.index(0.)]
    print("drug", drug217, "baseline", np.round(baseline217, 5).tolist())
    for scale217, response217 in zip(scales217, block217):
        print(scale217, np.round(response217 - baseline217, 5).tolist())

print("\n=== COMPACT HELPER INVENTORY ===")
for name217, obj217 in sorted(list(globals().items())):
    if (inspect.isfunction(obj217)
            and getattr(obj217, "__module__", "") == "__main__"
            and any(x in name217.lower() for x in
                    ["build", "make", "carrier", "posterior", "state_update", "readout", "dose"])):
        try:
            print(name217, inspect.signature(obj217))
        except Exception:
            pass

print("\n=== EARLY IDENTIFICATION NOTES (selected lines) ===")
lines217 = Path("/work/notes.md").read_text().splitlines()
shown217 = 0
for line217 in lines217[:220]:
    if any(x in line217.lower() for x in ["impulse", "instrument", "random", "kernel", "slope", "causal"]):
        print(line217[:380])
        shown217 += 1
        if shown217 >= 12:
            break

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 217; forwards 14. Testing installed physiological directions with equal signed fixed-history impulses, isolating downstream nonlinearity from generated-history feedback. No weights changed.\n")
```
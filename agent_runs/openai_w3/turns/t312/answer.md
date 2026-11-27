The architecture places normalization and recurrent feedback after the edited prelude, so raw output scaling need not scale the drug effect. The inspection was truncated before showing the retained helpers. I’ll retrieve those compactly to distinguish a saturated dose encoding from a downstream feedback problem.

```python
print("FUNCTION NAMES")
print([n for n, v in snapshot274 if inspect.isfunction(v) and not n.startswith("__")])

print("\nCHECKPOINTS")
for n, v in snapshot274:
    if n.startswith("checkpoint") and isinstance(v, dict):
        print(n, list(v)[:16])

print("\nSOURCE STRINGS")
for n, v in snapshot274:
    if isinstance(v, str) and len(v) > 500 and re.search(r"code|src|source|impl", n, re.I):
        print(n, len(v))

print("\nRELEVANT ARRAY NAMES")
for n, v in snapshot274:
    if hasattr(v, "shape") and re.search(r"193|222|267|273|carrier|direction|rate88|theta|beta|coef|target|ref", n, re.I):
        print(n, tuple(v.shape))

print("\nFACTUAL PASS 23")
for fn in sorted(os.listdir("/work/fwd_23")):
    if fn.startswith("factual") and fn.endswith(".csv"):
        d = pd.read_csv("/work/fwd_23/" + fn)
        print(fn, len(d), list(d.columns))
        print({c: float(d[c].mean()) for c in d.columns if re.search("nll|loss", c, re.I)})

print("\nNOTEBOOK RUNS 185-225")
lines275 = notes274.splitlines()
selected275 = [
    line for line in lines275
    if re.search(r"Python runs?: (18[5-9]|19\d|20\d|21\d|22[0-5])\b|Python run (18[5-9]|19\d|20\d|21\d|22[0-5])\b", line)
]
for line in selected275:
    print(line[:480])

print("\nEDIT SCALES")
for n in edited():
    a = load(n).detach().cpu().numpy()
    b = load_original(n).detach().cpu().numpy()
    print(n, tuple(a.shape), "original norm", round(float(np.linalg.norm(b)), 3),
          "delta norm", round(float(np.linalg.norm(a-b)), 3))

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 275; forwards 24. Architecture reread: edited prelude precedes normalization and recurrent feedback. Bounded recovery of implementation/checkpoints after run274 output truncation. No weights changed.\n")
```
The architecture confirms that a value-head edit leaves other heads’ weights untouched, though generated readings can still affect later predictions. The early estimates also show why raw dose–reading associations are unreliable. I’ll retrieve the final causal estimates and calibration frontier with shorter output; a useful candidate should reduce factual loss without sacrificing the supported responses.

```python
print("=== CAUSAL ESTIMATE FILES ===")
for fname283 in ["kernel_targets.csv", "dynamic_iv.csv", "sensor_iv.csv", "dml_results.csv"]:
    tab283 = pd.read_csv("/work/" + fname283)
    print(fname283, tab283.shape, list(tab283.columns))
    print(tab283.head(8).to_string(index=False))

print("=== CALIBRATION CANDIDATES ===")
front283 = pd.read_csv("/work/calibration_strength_frontier265.csv")
print(front283[(front283.insulin <= 0.90) &
               (front283.calibrator.isin(["soft_low", "soft_half"]))].to_string(index=False))

print("=== CONSTRUCTION VARIABLES ===")
items283 = list(globals().items())
selected283 = [(n, o) for n, o in items283
               if any(tag in n for tag in ["265", "268", "242", "247"])
               and not n.startswith(("name", "obj", "file", "table", "front", "fname", "tab"))]
for n283, o283 in selected283[:65]:
    if callable(o283):
        print(n283, "FUNCTION", o283.__code__.co_varnames[:o283.__code__.co_argcount],
              "refs", o283.__code__.co_names)
    elif isinstance(o283, dict):
        print(n283, "dict", list(o283)[:8])
    elif hasattr(o283, "shape"):
        print(n283, "shape", tuple(o283.shape))
    elif isinstance(o283, (list, tuple)):
        print(n283, type(o283).__name__, len(o283))
    elif isinstance(o283, (str, int, float)):
        print(n283, repr(o283)[:90])

print("=== NOTEBOOK CAUSAL SUMMARY EXCERPTS ===")
lines283 = notes282.splitlines()
matches283 = [i for i, line in enumerate(lines283)
              if ("target" in line.lower() or "identif" in line.lower())
              and ("kernel" in line.lower() or "causal" in line.lower()
                   or "insulin" in line.lower() or "norepinephrine" in line.lower())]
for i283 in matches283[:12]:
    print("\n".join(lines283[max(0, i283-1):i283+2])[:280])

with open("/work/notes.md", "a") as f283:
    f283.write("\n## Run 283: bounded causal and candidate recovery\n"
               "- Architecture and early estimates reconfirm treatment-selection confounding.\n"
               "- Retrieving final record-based estimates and calibration frontier; no edits changed.\n"
               "- Counts: 283 Python runs, 38 forward passes.\n")
```

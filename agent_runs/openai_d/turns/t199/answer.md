The architecture confirms that the current edit changes reading predictions, not dose selection. The early regressions also show that dose–outcome correlations are misleading, especially for norepinephrine. I’ll recover the later identification checks and Forward 22’s factual result; consistent adjusted effects and an NLL increase below 0.005 would justify further norepinephrine calibration.

```python
print("===== CAUSAL NOTEBOOK EXCERPT =====")
start = r171_notes.find("## Run 4")
print(r171_notes[start:start + 4700])

print("\n===== FORWARD 22 FACTUAL CHECK =====")
for filename in sorted(os.listdir("/work/fwd_22")):
    if filename.endswith(".csv"):
        frame = pd.read_csv("/work/fwd_22/" + filename)
        print(filename, "rows", len(frame), "columns", frame.columns.tolist())
        for col in frame.columns:
            if "nll" in col.lower() and pd.api.types.is_numeric_dtype(frame[col]):
                print(col, "mean", round(float(frame[col].mean()), 8))
                if "event" in frame:
                    print(frame.groupby("event")[col].mean().round(8).to_dict())

print("\n===== HELPER FUNCTIONS =====")
print(sorted(n for n, v in list(globals().items())
             if callable(v) and getattr(v, "__module__", None) == "__main__"))

print("\n===== RECENT VARIABLES =====")
for n in sorted(list(globals())):
    if re.match(r"r(165|166|167|168|169|170)_", n):
        v = globals()[n]
        if isinstance(v, dict):
            print(n, "keys", list(v)[:12])
        elif hasattr(v, "shape"):
            print(n, type(v).__name__, tuple(v.shape))
        else:
            print(n, repr(v)[:160])
```

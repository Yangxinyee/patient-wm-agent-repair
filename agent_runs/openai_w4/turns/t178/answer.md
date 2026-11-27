The recovered architecture shows that the dose response passes through recurrent state updates, so a stronger injection may amplify errors rather than produce the desired persistence. I’ll retrieve the numerical effect estimates and replay interfaces in a smaller output. A weak or rapidly decaying impulse response would support redesigning the connection; a sustained response would point elsewhere.

```python
import inspect, numpy as np, pandas as pd
from pathlib import Path

print("=== RECORD-BASED EFFECT ESTIMATES ===")
for name157 in ["causal_beta80", "kernels137", "provisional_kernels"]:
    obj157 = globals().get(name157, {})
    print(name157)
    for drug157, val157 in obj157.items():
        if isinstance(val157, dict):
            print(drug157, "keys", list(val157))
            for key157, vv157 in val157.items():
                if isinstance(vv157, (np.ndarray, list, tuple)):
                    aa157 = np.asarray(vv157)
                    if aa157.size <= 24:
                        print(key157, aa157)
                elif isinstance(vv157, (float, int, str, np.floating)):
                    print(key157, str(vv157)[:180])
        else:
            print(drug157, str(val157)[:450])
print("lp_new136 keys:",
      {k: list(v.keys()) if isinstance(v, dict) else type(v).__name__
       for k, v in lp_new136.items()})

print("\n=== BANK AND REPLAY INTERFACES ===")
print("bank:", {k: getattr(v, "shape", type(v).__name__) for k, v in repair_bank.items()})
for name157 in ["replay_factual125", "head_features_fixed", "transition73"]:
    fn157 = globals()[name157]
    print(name157, inspect.signature(fn157), "dependencies", fn157.__code__.co_names)
    print("constants", [x for x in fn157.__code__.co_consts
                        if isinstance(x, (str, int, float, tuple))][:25])
print("E65", E65.shape, "U65", U65.shape)
print("probe_data95:", type(probe_data95).__name__,
      list(probe_data95.columns) if hasattr(probe_data95, "columns") else "")
print("action/reading functions:",
      [(k, str(inspect.signature(v))) for k, v in list(globals().items())
       if inspect.isfunction(v) and any(s in k.lower() for s in
          ["action", "multiplier", "reading", "head_feature"])])
print("reading dataframes:",
      [(k, v.shape, list(v.columns)) for k, v in list(globals().items())
       if isinstance(v, pd.DataFrame) and any(s in k.lower() for s in
          ["read", "repair", "factual", "probe"])])

print("\n=== GPU ROLLOUT FILE SCHEMAS ===")
for folder157 in ["/work/fwd_7", "/work/fwd_10", "/work/fwd_11"]:
    for path157 in sorted(Path(folder157).glob("dose_edit*.csv")):
        df157 = pd.read_csv(path157)
        print(str(path157), df157.shape, list(df157.columns))
        print(df157.head(1).to_string(index=False))

print("\n=== EARLY IDENTIFICATION NOTES (COMPACT) ===")
shown157 = 0
for p157 in Path("/work/notes.md").read_text().split("\n## "):
    header157 = p157.split("\n", 1)[0]
    if any(s in p157.lower() for s in ["innovation", "identification", "kernel", "rollout"]):
        if not any(header157.startswith(f"Run {j}") for j in range(100, 158)):
            text157 = ("## " + p157)[:250]
            if shown157 + len(text157) <= 1800:
                print(text157)
                shown157 += len(text157)
Path("/work/notes.md").open("a").write(
    "\n## Run 157\nCompact recovery of record-based effect estimates and replay interfaces. "
    "No weights changed; recurrent persistence versus amplification still needs direct testing.\n")
```

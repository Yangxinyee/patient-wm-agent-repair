Halving the edited output path barely changed the rollout effects: insulin doubling still predicts too large a decrease, while halving predicts too little increase. This refutes simple output-scale calibration. I’ll recover the architecture and retained experiments before choosing another edit; a useful correction must change both dose responses while preserving factual fit.

```python
import os, inspect, re, pandas as pd, numpy as np
print("ARCHITECTURE")
print(open("/model/ARCHITECTURE.md").read())
print("\nDATA DOCUMENTATION")
print(open("/data/README.md").read())

notes274 = open("/work/notes.md").read()
print("\nEARLY NOTEBOOK")
print(notes274[:10000])

print("\nFORWARD 23 FACTUAL RESULTS")
for filename274 in sorted(os.listdir("/work/fwd_23")):
    if filename274.startswith("factual") and filename274.endswith(".csv"):
        df274 = pd.read_csv("/work/fwd_23/" + filename274)
        print(filename274, df274.shape, list(df274.columns))
        print(df274.select_dtypes(include="number").mean().to_dict())

snapshot274 = list(globals().items())
print("\nAVAILABLE FUNCTIONS")
for name274, value274 in snapshot274:
    if inspect.isfunction(value274) and not name274.startswith("__"):
        try:
            print(name274, str(inspect.signature(value274)))
        except Exception:
            print(name274)

print("\nRETAINED VARIABLES")
for name274, value274 in snapshot274:
    if re.search(r"checkpoint|carrier|direction|target|reference|267|273|193|222|rate88|theta|beta|coef|code|source", name274, re.I):
        if hasattr(value274, "shape"):
            description274 = str(value274.shape)
        elif isinstance(value274, dict):
            description274 = "dict keys=" + str(list(value274)[:30])
        elif isinstance(value274, str):
            description274 = "str length=" + str(len(value274))
        elif isinstance(value274, (list, tuple)):
            description274 = type(value274).__name__ + " length=" + str(len(value274))
        else:
            description274 = type(value274).__name__
        print(name274, description274)

with open("/work/notes.md", "a") as file274:
    file274.write("\nPython run 274; forwards 24. Restart recovery: whole-output-path halving scarcely changed rollout effects (-19.559,+1.026,+1.639,-1.328), refuting simple output-scale calibration. Recovering architecture, early identification evidence, helpers and factual pass 23. No weights changed.\n")
```
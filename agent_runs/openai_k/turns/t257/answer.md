The rebuilt replay matches official states closely, and the dose encoder reproduces factual action vectors exactly. Doubled doses change the prelude features, but only weakly change the normalized output—evidence of attenuation downstream. I’ll check the value-head replay and gather factual control features before fitting a low-rank correction. The correction must amplify dose contrasts without adding substantial factual noise.

```python
from pathlib import Path
import pandas as pd
import numpy as np

print("=== GENERIC CORE CONTEXT CONSTRUCTION ===")
print(compact227(r59_core)[:3500])

print("=== VALUE HEAD, REMAINING INSTRUCTIONS ===")
head231 = compact227(r49_readouts)
start231 = head231.find("\n33:")
print(head231[start231:][:3900])

print("=== AVAILABLE FACTUAL REFERENCES ===")
facts231 = []
for p231 in sorted(Path("/work").glob("fwd_*/factual_original.csv")):
    t231 = pd.read_csv(p231, nrows=2)
    facts231.append((str(p231), list(t231.columns), t231.iloc[0].to_dict()))
for item231 in facts231:
    print(item231[0], str(item231[2])[:210])
if facts231:
    print("columns:", facts231[0][1])

def original_token231(u):
    z = mlp0(torch.cat([context151["ctx"], context151["e"], u], -1),
             "z_post")[:, :32]
    return linear228(z, "z_in")

original_cf231 = {}
for d231 in ("insulin", "norepinephrine"):
    u231 = actions230[d231, 2.0]
    original_cf231[d231] = core228(original_token231(u231), u231)
    print("original", d231, "E dose RMS",
          float((original_cf231[d231]["E"]-base228["E"]).square().mean().sqrt()))
print("base mean-E covariance range:",
      np.quantile(base228["E"].mean(1).numpy().std(0), [0,0.25,0.5,0.75,1]))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 231: prepare constrained recurrent correction\n"
            "- Core replay validated: output RMS 1.38e-5; h RMS .00122; posterior RMS 1.57e-5.\n"
            "- Factor-one action encoding is exact. Dose changes reach E but are attenuated in out.\n"
            "- Recovering generic replay and head reference to constrain factual side effects.\n"
            "- No weights changed. Python runs: 231; forward passes: 17.\n")
```
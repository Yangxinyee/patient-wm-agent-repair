The corrected time decay now matches the verified replay exactly. The cached slot residuals differ by about 23.7 RMS, confirming a substantial fitting-cache error. The diagnostic then stopped on a NumPy/Torch conversion, without changing weights. I’ll finish the alignment checks and locate the counterfactual action vectors; accurate posterior and prelude agreement would support refitting.

```python
import numpy as np, torch

print("=== REMAINING ALIGNMENT CHECKS ===")
posterr278 = muorig277.numpy()-np276(rr_zexport4)
print("original posterior RMS, valid:",
      float(np.sqrt(np.mean(posterr278[same277]**2))))
print("original posterior RMS, hour >= 0:",
      float(np.sqrt(np.mean(posterr278[hours276 >= 0]**2))))
check278 = torch.as_tensor(np276(raw272), dtype=torch.float32)
oldcheck278 = prelude269(check278)
wx278 = {n: torch.as_tensor(np276(rx_W[n])) for n in NAMES}
newcheck278 = prelude277(check278, wx278)
print("independent prelude versus exact helper RMS:",
      float(torch.sqrt(torch.mean((oldcheck278-newcheck278)**2))))
print("corrected E RMS:", float(np.sqrt(np.mean(Ef277**2))))
pold278 = np276(fits266[1.0])
print("old dose projection on corrected E RMS:",
      np.sqrt(np.mean((Ef277.reshape(-1, 192) @ pold278)**2, axis=0)))

print("\n=== COUNTERFACTUAL CACHE DICTIONARIES ===")
for n in sorted(list(globals())):
    o = globals()[n]
    if isinstance(o, dict) and ("insulin", 2.0) in o:
        val = o[("insulin", 2.0)]
        print(n, "keys", list(o.keys())[:3],
              "example", type(val).__name__, getattr(val, "shape", None))
    elif isinstance(o, dict) and "insulin" in o and re.search(r"(ds_|rate|action|^r2)", n):
        print(n, "drug-key dict", [(k, getattr(v, "shape", None)) for k,v in o.items()][:5])

print("\n=== PHYSICAL DOSE OBJECTS ===")
for n in sorted(list(globals())):
    if n.startswith("ds_"):
        o = globals()[n]
        if isinstance(o, dict):
            print(n, "dict", [(k, getattr(v, "shape", None)) for k,v in o.items()][:6])
        elif hasattr(o, "shape"):
            print(n, tuple(o.shape))
        elif callable(o) and hasattr(o, "__code__"):
            print(n, "fn", o.__code__.co_varnames[:o.__code__.co_argcount],
                  "uses", o.__code__.co_names)

print("\n=== ACTION DESCRIPTORS ===")
print("rr_desc4 examples", rr_desc4[:3])
idx278 = np.flatnonzero(np276(rr_act_mask112))[:3]
for i in idx278:
    print("action", int(i), rr_desc4[int(i)])
print("action encoder constants", rr_encode_atoms112.__code__.co_consts)
print("records columns", list(rr_records4.columns))

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 277–278\nCorrected rounded-minute hflow matches verified replay exactly; old cached slot residuals differ by RMS23.74. Run277 stopped on a NumPy/Torch conversion after building finite corrected features. Completing posterior/prelude checks and recovering counterfactual action caches before fitting. No weights changed. Counts: 278 Python runs, 29 forwards.\n")
```
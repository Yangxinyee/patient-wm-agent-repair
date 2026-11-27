The neutral channel passed the matched factual test, but direct token calibration was weak because many states ignored that token. The custom attention route caused much larger damage. I’ll check whether its response signal is truly zero on untreated bundles, including admission. Material leakage there would refute the gating design before any further dose calibration.

```python
print("=== ATTENTION CONTRACTS ===")
for nm in ("attention136", "att_features220", "core_step136", "encode_actions168"):
    fn = globals()[nm]
    print(nm, inspect.signature(fn), "names", fn.__code__.co_names,
          "constants", [x for x in fn.__code__.co_consts if isinstance(x,(str,int,float,tuple))][:20])
print("config", VOCAB.get("config"))

print("\n=== DIRECT TOKEN DIRECTION NORMS ===")
for nm in ("mean_directions216", "candidates216", "directions231"):
    v = globals()[nm]
    print(nm, {str(k): round(float(np.linalg.norm(array196(x))), 4) for k,x in v.items()})
print("private coordinates",
      [(nm, str(v)[:150]) for nm,v in list(globals().items())
       if ("coord" in nm.lower() or "response_col" in nm.lower())
       and isinstance(v,(int,list,tuple,np.ndarray))][:15])

print("\n=== ROUTING / DIRECT-TEST RETAINED VARIABLES ===")
for nm,v in list(globals().items()):
    if nm.endswith(("216","222","237","238","243")):
        if isinstance(v,(np.ndarray,torch.Tensor)):
            print(nm, tuple(v.shape),
                  str(array196(v))[:120] if v.numel() < 12 and isinstance(v,torch.Tensor)
                  else (str(v)[:120] if isinstance(v,np.ndarray) and v.size < 12 else ""))
        elif isinstance(v,dict):
            print(nm, "dict", list(v)[:8])
        elif isinstance(v,(int,float,list,tuple,str)) and not inspect.isfunction(v):
            print(nm, str(v)[:120])

print("\n=== ADMISSION COVERAGE ===")
for nm in ("hours156","hours","input196","S166","stays156"):
    v = globals().get(nm)
    if isinstance(v,np.ndarray):
        print(nm, v.shape)
        if v.ndim == 1 and ("hour" in nm):
            print("min", v.min(), "negative count", int((v<0).sum()),
                  "hour-zero count", int((v==0).sum()))
print("training admission rows")
tr250 = pd.read_csv("/data/train.csv", nrows=65)
print(tr250[["stay_id","hour","event","rate"]].head(12).to_string(index=False))

print("\n=== GATING ERROR BY HOUR ===")
if "innovation237" in globals() and "inactive237" in globals():
    a = np.asarray(innovation237)
    mask = np.asarray(inactive237, dtype=bool)
    hh = np.asarray(hours156)
    for label,m in (("negative", hh<0),("zero",hh==0),("later",hh>0)):
        m = m & mask
        print(label, int(m.sum()),
              np.sqrt(np.mean(a[m]**2,axis=0)) if m.any() else None,
              np.max(np.abs(a[m]),axis=0) if m.any() else None)

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 250 — audit untreated routing and startup coverage\n"
            "- Neutral209 factual predictions matched original on the first 100 tuning stays.\n"
            "- Direct-token calibration previously underdelivered; custom attention response failed factual fit.\n"
            "- Checking untreated innovation leakage and startup coverage before another edit; no tensors changed.\n")
```
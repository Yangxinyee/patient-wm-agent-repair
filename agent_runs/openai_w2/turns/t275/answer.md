The inventory scan failed before any weights changed because it treated a function as an array. I’ll fix the scan and rerun the blend checks. The conversion must reproduce the existing decoder accurately before any new trial is installed.

```python
print("193-COLUMN FACTUAL DESIGNS:",
      [(n, tuple(v.shape)) for n,v in list(globals().items())
       if isinstance(getattr(v,"shape",None),tuple)
       and len(v.shape)==2 and v.shape[1]==193 and v.shape[0]>100
       and not n.endswith("226")][:35])

coef_linear226 = np.asarray(
    constrained_fits206[(("linear_half","preserve_cross"),(0.003,1e-6))],
    dtype=np.float64)
coef_base226 = np.asarray(coef212, dtype=np.float64)
print("BASE FIT CONSISTENCY:",
      np.max(np.abs(coef_base226 - np.asarray(
          constrained_fits206[(("preserve_half","preserve_cross"),(0.003,1e-6))]))))

E226 = np.asarray(raw_E212, dtype=np.float64)
X226 = np.asarray(design212, dtype=np.float64)
sd226 = E226.std(axis=0) / X226[:,1:].std(axis=0)
mu226 = E226.mean(axis=0) - sd226*X226[:,1:].mean(axis=0)
print("STANDARDIZATION ERROR:",
      np.max(np.abs((E226-mu226)/sd226-X226[:,1:])))

gamma226 = np.asarray(ln_gamma212, dtype=np.float64)
beta226 = np.asarray(ln_beta212, dtype=np.float64)
cd226 = 1.0/gamma226
ck226 = float(cd226@beta226)
print("LAYERNORM CONSTANT:", ck226)

def decoder_to_weight226(coef):
    linear = np.asarray(coef)[1:] / sd226
    intercept = float(np.asarray(coef)[0] - mu226@linear)
    return linear + (intercept/ck226)*cd226

mapped_base226 = decoder_to_weight226(coef_base226)
print("EXISTING DECODER WEIGHT ERROR:",
      np.max(np.abs(mapped_base226-np.asarray(candidate_W212)[:,0])))
print("EXISTING RESPONSE MAPPING ERROR:",
      np.max(np.abs(E226@mapped_base226-X226@coef_base226)))
assert np.max(np.abs(mapped_base226-np.asarray(candidate_W212)[:,0])) < 1e-5

blend_rows226 = []
for alpha226 in [0.0,0.25,0.5,0.75,1.0]:
    bc226 = (1-alpha226)*coef_base226 + alpha226*coef_linear226
    responses226 = {k:float(np.asarray(v)@bc226) for k,v in probe_means206.items()}
    blend_rows226.append({
        "linear_half_fraction":alpha226,
        "old_factual_RMS":float(np.sqrt(np.mean((X226@bc226)**2))),
        "insulin_double":responses226[("insulin",2.0)],
        "insulin_half":responses226[("insulin",0.5)],
        "NE_double_cross":responses226[("norepinephrine",2.0)],
        "mapping_error":float(np.max(np.abs(
            E226@decoder_to_weight226(bc226)-X226@bc226)))
    })
blend_table226 = pd.DataFrame(blend_rows226)
print("\nPARTIAL-BLEND PROXIES (not NLL):")
print(blend_table226.to_string(index=False))

with open("/work/notes.md","a") as f227:
    f227.write("\n## Runs226–227: half-dose decoder blend diagnostics\n")
    f227.write("Run226 inventory error occurred before any edits. No weight changes in either run.\n")
    f227.write("Verified standardization and LayerNorm-to-weight mapping.\n")
    f227.write(blend_table226.to_string(index=False)+"\n")
    f227.write("Residual RMS is only a disturbance proxy; any trial needs direct paired factual NLL and dose rollouts.\n")
```

The recovered fits suggest that strengthening the half-dose response has a factual-disturbance cost. I’ll measure partial blends and verify their conversion into the weight tensor before choosing a trial. A modest residual increase with an exact conversion would justify testing; it would not itself establish that factual NLL passes.

```python
import inspect

print("FIT FUNCTION:", inspect.signature(constrained_small_fit206))
print("FIT DEPENDENCIES:", constrained_small_fit206.__code__.co_names)
print("193-COLUMN FACTUAL DESIGNS:",
      [(n, tuple(v.shape)) for n,v in list(globals().items())
       if hasattr(v,"shape") and len(v.shape)==2 and v.shape[1]==193
       and v.shape[0]>100 and not n.endswith("226")][:35])

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
    responses226 = {str(k):float(np.asarray(v)@bc226) for k,v in probe_means206.items()}
    blend_rows226.append({
        "linear_half_fraction":alpha226,
        "old_factual_RMS":float(np.sqrt(np.mean((X226@bc226)**2))),
        "insulin_double":responses226[str(("insulin",2.0))],
        "insulin_half":responses226[str(("insulin",0.5))],
        "NE_double_cross":responses226[str(("norepinephrine",2.0))],
        "mapping_error":float(np.max(np.abs(
            E226@decoder_to_weight226(bc226)-X226@bc226)))
    })
blend_table226 = pd.DataFrame(blend_rows226)
print("\nPARTIAL-BLEND PROXIES (not NLL):")
print(blend_table226.to_string(index=False))

with open("/work/notes.md","a") as f226:
    f226.write("\n## Run226: half-dose decoder blend diagnostics\n")
    f226.write("No weight changes. Verified standardization and LayerNorm-to-weight mapping.\n")
    f226.write(blend_table226.to_string(index=False)+"\n")
    f226.write("Residual RMS is only a disturbance proxy; any installed trial needs direct paired factual NLL and dose rollouts.\n")
```

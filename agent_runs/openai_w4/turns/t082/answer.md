The retained immediate estimates are about −2.05 glucose units per insulin-rate unit and +3.17 MAP units per norepinephrine-rate unit. The original model’s rollout shifts are much smaller, and the lag moments suggest persistence. I’ll inspect the deconvolved responses and decoder checks next; a correction is credible only if the response estimates and state decoding hold beyond the fitting stays.

```python
print("PROVISIONAL KERNELS:", provisional_kernels)
print("PERSISTENCE FITS:", persistence_repair)
for drug, tb in impulse_tables_repair.items():
    print("\nIMPULSE", drug)
    print(tb.to_string(index=False))
print("\nHEAD CONFIG:", head_cfg_repair)
print("HEAD INNOVATION PROBE SUMMARY")
for k, v in head_innovation_probes.items():
    if isinstance(v, dict):
        print(k, {a: b for a, b in v.items()
                  if np.isscalar(b) or isinstance(b, str)})
    elif isinstance(v, pd.DataFrame):
        print(k, v.to_string(index=False))
    else:
        print(k, type(v).__name__, np.shape(v))
print("\nDECODER SUMMARY")
for n in ("insulin_decoder_repair", "decoder_candidates_repair"):
    vv = globals()[n]
    if isinstance(vv, dict):
        print(n, {k: v for k, v in vv.items()
                  if np.isscalar(v) or isinstance(v, str)})
        print("keys:", list(vv.keys()))
    else:
        print(n, type(vv).__name__, np.shape(vv))
print("\nPOLICY MODEL STRUCTURE")
for drug, md in dml_models.items():
    print(drug, [(k, type(v).__name__) for k, v in md.items()]
          if isinstance(md, dict) else type(md).__name__)
print("\nINITIAL NOTES")
print(open("/work/notes.md").read()[:5200])
```

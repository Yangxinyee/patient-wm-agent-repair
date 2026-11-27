The gate can suppress the large responses on irrelevant sensors, but the norepinephrine decoder still has substantial error on MAP features. Some one-step target shifts also look unexpectedly large. I’ll audit their dose units before fitting anything stronger. If the stored contrasts use different units from the record-based coefficients, the targets need rescaling.
```python
print("=== RAW DRUG RATES ===")
for drug in ["insulin", "norepinephrine"]:
    s = train.loc[train.event == drug, "rate"].dropna()
    print(drug, "n", len(s), "quantiles",
          s.quantile([0,.01,.1,.5,.9,.99,1]).to_dict())
    for fac in [.5, 2.0, 3.0]:
        c = cf_features81[(drug, fac)]
        r = np.asarray(c["rows"])
        d = np.asarray(c["delta_rate"])
        m = np.isin(events179[r],
                    ["glucose_finger", "glucose_lab"] if drug == "insulin"
                    else ["map_arterial", "map_cuff"])
        print(" ", fac, "relevant contrast quantiles",
              np.quantile(d[m], [0,.1,.5,.9,1]).round(5).tolist(),
              "mean", float(d[m].mean()))
        print("  first rows", r[:8].tolist(), "delta", d[:8].tolist())

print("\n=== EFFECT / UNIT VARIABLE INVENTORY ===")
for n in sorted(list(globals())):
    if any(s in n.lower() for s in ["rate81", "rates81", "effect19", "kernel19",
                                  "kernels19", "panel", "drug_rows", "cases81",
                                  "effect14", "effect15", "target97", "target19"]):
        describe196(n, globals()[n])

print("\n=== CAUSAL KERNEL NOTES ===")
text198 = Path("/work/notes.md").read_text()
for marker in ["## Run 18", "## Run 19", "## Run 20", "## Run 21"]:
    pos = text198.find(marker)
    if pos >= 0:
        nxt = text198.find("\n## ", pos + 5)
        print(text198[pos:nxt if nxt >= 0 else pos + 1800][:1800])

print("\n=== SAMPLE / CONTRAST HELPERS ===")
for n in sorted(list(globals())):
    f = globals()[n]
    if callable(f):
        c = getattr(f, "__code__", None)
        if c is not None and (any("cf_features" in s for s in c.co_names) or
                              any(s in c.co_names for s in ["delta_rate", "rate81"])):
            print(n, "args", c.co_varnames[:c.co_argcount],
                  "locals", c.co_varnames[:30], "names", c.co_names[:35],
                  "constants", tuple(x for x in c.co_consts
                                     if isinstance(x, (str, int, float, tuple))))
print("\nREADME")
print(Path("/data/README.md").read_text()[:3500])
```

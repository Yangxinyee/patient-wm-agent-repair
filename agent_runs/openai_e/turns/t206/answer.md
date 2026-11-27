The offsets modestly reduce divergence, but they do not consistently improve held-out NLL. I won’t install them. I’ll review the measured rollout effects and factual costs of the saved candidates to identify the remaining gap. Any next edit must improve the dose readouts while retaining the factual margin.

```python
print("DOSE-EDIT RESULTS ON DISK")
dosefiles177 = sorted(Path("/work").glob("fwd_*/dose_edit_*.csv"),
                      key=lambda p: (int(p.parent.name.split("_")[1]), p.name))
for i177, file177 in enumerate(dosefiles177):
    df177 = pd.read_csv(file177)
    if i177 == 0:
        print("COLUMNS", list(df177.columns))
        print(df177.head(2).to_string(index=False))
    keys177 = [c for c in ("edit", "drug", "factor", "multiplier", "outcome", "target")
               if c in df177.columns]
    value177 = next((c for c in ("mean_shift", "shift", "delta", "predicted_shift")
                     if c in df177.columns), None)
    print(str(file177.relative_to("/work")), "rows", len(df177))
    if keys177 and value177:
        print(df177.groupby(keys177, dropna=False)[value177].mean().to_string())
    else:
        print(df177.head(1).to_string(index=False))

print("\nFACTUAL RESULTS")
for file177 in sorted(Path("/work").glob("fwd_*/factual_*.csv"),
                      key=lambda p: (int(p.parent.name.split("_")[1]), p.name)):
    df177 = pd.read_csv(file177)
    if "nll" in df177:
        print(str(file177.relative_to("/work")), len(df177),
              "NLL", round(float(df177.nll.mean()), 7))

print("\nCANDIDATE HISTORY, BOUNDED")
selected177 = [s for s in nt170.splitlines()
               if re.search(r"Forward|NLL=|rollout shifts|readouts|dose shifts", s, re.I)]
for line177 in selected177[-18:]:
    print(line177[:280])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 177: do not install weak calibration; audit measured candidates\n"
            "Code/last-bin distillation offsets slightly reduce KL but held-out NLL does not consistently improve. "
            "No decoder edit installed. Reviewing on-disk factual and rollout results to select the next "
            "targeted experiment. Python runs: 177; forwards: 22.\n")
```
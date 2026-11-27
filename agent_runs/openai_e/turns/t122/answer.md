The recovered analyses support norepinephrine targets around +4.2 to +4.6 for doubling and −2.1 to −2.3 for halving. The insulin baseline predicts only +0.10 and −1.00 for those interventions. I’ll inspect the saved correction and matched factual results to choose a reduced trial; its benefit must justify the limited NLL headroom.

```python
import inspect

print("SAVED TRIAL")
trial106 = np.load("/work/ne_trial103.npz")
print([(k, trial106[k].shape) for k in trial106.files])

print("RECENT OBJECTS")
for k106, v106 in list(globals().items()):
    if k106.endswith(("95", "100", "102", "103")) and not k106.startswith("_"):
        if hasattr(v106, "shape"):
            print(k106, tuple(v106.shape))
        elif isinstance(v106, dict):
            print(k106, "keys", list(v106)[:5])
        elif isinstance(v106, (int, float, str)):
            print(k106, str(v106)[:90])

print("BASELINE FACTUAL AND TRIAL COMPARISON")
for folder106 in ("fwd_9", "fwd_11", "fwd_13"):
    pair106 = {}
    for model106 in ("original", "edited"):
        file106 = Path("/work") / folder106 / f"factual_{model106}.csv"
        if file106.exists():
            df106 = pd.read_csv(file106)
            pair106[model106] = df106
            print(folder106, model106, df106.shape, df106.columns.tolist())
            nllcols106 = [c for c in df106.columns if "nll" in c.lower()]
            print("NLL", {c: float(df106[c].mean()) for c in nllcols106})
            staycols106 = [c for c in df106.columns if "stay" in c.lower()]
            if staycols106:
                print("STAY SET", df106[staycols106[0]].nunique(),
                      sorted(df106[staycols106[0]].unique())[:3])
print("ROLLOUT SAMPLE")
df106 = pd.read_csv("/work/fwd_10/dose_edit_edited.csv")
print(df106.columns.tolist())
print(df106.head(3).to_string(index=False))
print("HELPER SIGNATURES")
for name106 in ("prelude46", "coda46", "next_means45", "head_calc",
                "core_update_local", "posterior94", "full_insulin_slope58"):
    print(name106, inspect.signature(globals()[name106]))

print("INSULIN EDIT NOTES")
for sec106 in sections105:
    m106 = re.search(r"Run\s+(\d+)", sec106.splitlines()[0])
    if m106 and int(m106.group(1)) in (58, 63, 64):
        print(sec106[:650])
with Path("/work/notes.md").open("a") as f106:
    f106.write("\n## Run 106: recover matched baseline and trial parameters\n"
               "Record-derived NE target remains roughly +4.2..4.6/-2.1..2.3; "
               "baseline NE predictions +.099/-1.004. Inspecting saved trial and "
               "matched factual files to choose a smaller reversible edit. "
               "Python runs: 106; forwards: 13.\n")
```
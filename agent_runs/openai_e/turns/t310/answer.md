The reduced prelude edit fails factual safety: NLL increased by 0.005713, above the 0.005 limit. I’ll remove it and restore the previous candidate, then review earlier norepinephrine trials. A useful next edit must strengthen norepinephrine responses without spending more factual-fit margin than we have.

```python
drop_edit("core.prelude.0.ffn.2.weight")
drop_edit("core.prelude.0.ffn.2.bias")
with np.load("/work/joint236.npz") as a:
    joint266 = {k: a[k].copy() for k in a.files}
save_edit("core.b_in.weight",
          load_original("core.b_in.weight") + joint266["delta"])
print("RESTORED joint236; edits:", edited())

notes266 = Path("/work/notes.md").read_text()
print("\n=== NOREPINEPHRINE CANDIDATE HISTORY ===")
for run in (82, 107, 148, 151, 152, 154, 155, 199, 200, 208, 210):
    ms = list(re.finditer(rf"(?m)^##[^\n]*\b{run}\b[^\n]*", notes266))
    for m in ms[-1:]:
        end = notes266.find("\n##", m.end())
        print(notes266[m.start():end if end >= 0 else None][:620])

print("\n=== FORWARD OUTPUT TABLE ===")
for k in range(1, 33):
    folder = Path(f"/work/fwd_{k}")
    if not folder.exists():
        continue
    for model in ("original", "edited"):
        fp = folder/f"factual_{model}.csv"
        if fp.exists():
            d = pd.read_csv(fp)
            print(k, model, "factual", len(d), float(d["nll"].mean()))
        dp = folder/f"dose_edit_{model}.csv"
        if dp.exists():
            d = pd.read_csv(dp)
            if k == 1 and model == "original":
                print("DOSE COLUMNS", d.columns.tolist(), d.head(1).to_dict("records"))
            ec = next((c for c in ("edit", "intervention", "dose_edit") if c in d), None)
            sc = next((c for c in ("shift", "mean_shift", "delta") if c in d), None)
            if ec and sc:
                reached = d["reached"].astype(str).str.lower().isin(["true", "1", "1.0"]) if "reached" in d else np.ones(len(d), bool)
                print(k, model, "dose", d.loc[reached].groupby(ec)[sc].mean().round(3).to_dict())

print("\n=== NE ARCHIVE CONTRACTS ===")
for fn in ("ne_candidate151.npz", "ne_candidate199.npz", "ne_candidate82.npz"):
    with np.load("/work/"+fn) as a:
        print(fn, [(k, a[k].shape) for k in a.files])
        print({k: a[k].tolist() for k in a.files if a[k].size <= 8})

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 266: reject reduced nonlinear edit\n"
            "Forward31 factual cost=.00571319, exceeding the .005 limit. "
            "Dropped both prelude edits and restored exact joint236 b_in. "
            "Reviewing earlier NE trials and factual/rollout frontier before another trial.\n"
            "Python runs: 266; forwards: 32.\n")
```
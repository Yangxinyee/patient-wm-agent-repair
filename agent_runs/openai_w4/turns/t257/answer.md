The carriers use paired neurons with large gating weights, so changing a single bias could damage factual predictions. I’ll first compare the earlier validated rollouts and measure the gate activations. If an insulin-only candidate had better symmetry, that would favor reverting the dual-channel extension rather than adding more complexity.

```python
print("=== ALL FORWARD ROLLOUT READOUTS: ELIGIBLE, EQUAL-STAY MEANS ===")
rows224 = []
for folder in sorted(Path("/work").glob("fwd_*"),
                     key=lambda p: int(p.name.split("_")[-1])):
    for model in ["original", "edited"]:
        path = folder / f"dose_edit_{model}.csv"
        if not path.exists():
            continue
        d = pd.read_csv(path)
        ok = d["reached"].astype(str).str.lower().eq("true")
        d = d.loc[ok & d["hour"].between(13, 24)]
        means = d.groupby(["edit", "stay_id"])["shift"].mean().groupby("edit").mean()
        row = {"pass": folder.name, "model": model, "stays": d.stay_id.nunique()}
        row.update(means.round(3).to_dict())
        rows224.append(row)
print(pd.DataFrame(rows224).to_string(index=False))

print("\n=== PRE-DUAL INSULIN REPAIR NOTES ===")
budget224 = 3500
for section in Path("/work/notes.md").read_text().split("\n## "):
    heading = section.splitlines()[0] if section else ""
    m = re.match(r"Run(?:s)?\s+(\d+)", heading)
    if m and 159 <= int(m.group(1)) <= 180 and budget224 > 0:
        text = section[:min(500, budget224)]
        print("##", text)
        budget224 -= len(text)

print("\n=== PAIRED-NEURON GEOMETRY ===")
for pair in [(96, 184), (89, 156)]:
    r1, r2 = pair
    print(pair, "sum norms", np.round([
        np.linalg.norm(a223[r1, j:j+192] + a223[r2, j:j+192])
        for j in [0, 192, 384]], 4),
        "difference norms", np.round([
        np.linalg.norm(a223[r1, j:j+192] - a223[r2, j:j+192])
        for j in [0, 192, 384]], 4))
print("raw_base192", type(raw_base192).__name__, getattr(raw_base192, "shape", None))
raw224 = as_np184(raw_base192)
if raw224.ndim == 2 and raw224.shape[1] in (576, 577):
    x224 = raw224[:, :576]
    pre224 = x224 @ a223[rows223].T + as_np184(cur223["z_post.0.bias"])[rows223]
    print("all-bundle preactivation quantiles, rows", rows223.tolist())
    print(np.round(np.quantile(pre224, [.01, .1, .5, .9, .99], axis=0), 3))
    print("active192 shape", np.asarray(active192).shape)
    mask224 = np.asarray(active192, dtype=bool)
    if mask224.shape == (len(x224),):
        print("NE-bundle preactivation quantiles")
        print(np.round(np.quantile(pre224[mask224], [.01, .1, .5, .9, .99], axis=0), 3))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 224\nComparing all retained free-rollout candidates using eligible-hour equal-stay readouts "
            "and inspecting paired-neuron gate geometry before nonlinear surgery. "
            "No saved weights changed; checking whether dual-channel construction introduced insulin asymmetry.\n")
```

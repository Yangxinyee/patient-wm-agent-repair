The fresh-test stays are uniquely identified, and a paired result already exists in `fwd_36`. I’ll inspect that result before spending another forward pass. If the reverted baseline still exceeds the loss limit, persistence or centering—not only the rejected strength increase—needs revision.

```python
import json
from safetensors.torch import load_file

print("LATEST FORWARD SUMMARIES")
for k in (33, 34, 35, 36):
    p = Path(f"/work/fwd_{k}")
    if (p / "summary.json").exists():
        print(k, (p / "summary.json").read_text()[:1100])
    fo, fe = p / "factual_original.csv", p / "factual_edited.csv"
    if fo.exists() and fe.exists():
        o, e = pd.read_csv(fo), pd.read_csv(fe)
        assert o[["stay_id", "hour", "event"]].equals(
            e[["stay_id", "hour", "event"]])
        paired = e[["stay_id", "event"]].copy()
        paired["delta_nll"] = e.nll.to_numpy() - o.nll.to_numpy()
        bystay = paired.groupby("stay_id").delta_nll.mean()
        print("NLL original / edited / delta:",
              o.nll.mean(), e.nll.mean(), paired.delta_nll.mean())
        print("By event:", paired.groupby("event").delta_nll.mean().to_dict())
        print("Stay-level delta mean / SE:",
              bystay.mean(), bystay.std(ddof=1) / np.sqrt(len(bystay)))

print("\nCHECKPOINT CONSISTENCY")
base292 = load_file("/work/checkpoint_run289.safetensors")
diff292 = {}
for name, old in base292.items():
    current = load(name)
    current_np = (current.detach().cpu().numpy()
                  if hasattr(current, "detach") else np.asarray(current))
    err = float(np.max(np.abs(current_np - old.numpy())))
    if err > 1e-6:
        diff292[name] = err
print("Current differences from checkpoint_run289:", diff292)

print("\nCONTROL VARIABLE NAMES")
control_names292 = sorted(
    k for k in list(globals())
    if not k.startswith("_") and any(s in k.lower() for s in
       ("base289", "signal", "center", "private", "beta", "target", "strength"))
)
print(control_names292[:110])

print("\nEARLY EVIDENCE NOTES")
print(Path("/work/notes.md").read_text()[:3500])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 292: compact fresh-stay acceptance audit\n"
            "- Inspected paired factual results in forwards 35 and 36 and checked "
            "current weights against checkpoint_run289.\n"
            "- No weights changed; decisions will use the paired losses, not the "
            "failed duplicate-ID request.\n")
```
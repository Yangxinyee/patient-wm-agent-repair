The inspection stopped on a non-array object; it did not change the restored baseline. I’ll verify how the saved trial is composed and install 35% of its norepinephrine correction. This tests whether a reduced edit can retain a useful dose response while staying within the factual NLL limit.

```python
import torch

base107 = np.asarray(archive104["delta"], dtype=np.float32)
ne107 = (float(trial106["strength"]) *
         np.outer(trial106["direction"], trial106["feature"])).astype(np.float32)
saved107 = np.asarray(trial106["delta"], dtype=np.float32)
err_total107 = float(np.linalg.norm(saved107 - base107 - ne107))
err_ne107 = float(np.linalg.norm(saved107 - ne107))
print("ARCHIVE CHECK", {"total_delta_error": err_total107,
                        "NE_only_error": err_ne107,
                        "NE_norm": float(np.linalg.norm(ne107))})
assert min(err_total107, err_ne107) < 0.01, "Saved trial composition needs inspection"

pairs107 = {}
for folder107 in ("fwd_9", "fwd_13"):
    pairs107[folder107] = {}
    for model107 in ("original", "edited"):
        path107 = Path("/work") / folder107 / f"factual_{model107}.csv"
        frame107 = pd.read_csv(path107)
        pairs107[folder107][model107] = frame107
        nll107 = [c for c in frame107.columns if "nll" in c.lower()]
        print(folder107, model107, "columns", frame107.columns.tolist(),
              "NLL", {c: round(float(frame107[c].mean()), 6) for c in nll107})
staycol107 = next(c for c in pairs107["fwd_13"]["original"].columns if "stay" in c.lower())
ids107 = sorted(pairs107["fwd_13"]["original"][staycol107].unique().tolist())
ids9_107 = set(pairs107["fwd_9"]["original"][staycol107].unique())
print("SAME VALIDATION STAYS", ids9_107 == set(ids107))

roll107 = pd.read_csv("/work/fwd_10/dose_edit_edited.csv")
print("ROLLOUT COLUMNS", roll107.columns.tolist())
print(roll107.head(2).to_string(index=False))
for sec107 in sections105:
    m107 = re.search(r"Run\s+(\d+)", sec107.splitlines()[0])
    if m107 and int(m107.group(1)) in (58, 63, 76):
        print(sec107[:1000])

scale107 = 0.35
candidate107 = base107 + scale107 * ne107
original107 = load_original("core.b_in.weight")
save_edit("core.b_in.weight",
          original107 + torch.as_tensor(candidate107, dtype=original107.dtype)
          if isinstance(original107, torch.Tensor) else original107 + candidate107)
np.savez("/work/ne_reduced107.npz", delta=candidate107, ne_delta=ne107,
         scale=scale107)
print("INSTALLED", {"NE_scale": scale107, "edits": edited()})
with Path("/work/notes.md").open("a") as f107:
    f107.write("\n## Run 107: reduced corrected NE trial\n"
               "Run106 inventory errored on a non-array; weights were unchanged. "
               f"Verified trial103 composition (total error {err_total107:.6g}, "
               f"NE-only error {err_ne107:.6g}). Installed baseline77 plus .35 times "
               "trial103 NE delta; unaccepted pending paired factual and rollout tests. "
               "Python runs: 107; forwards: 13.\n")
```
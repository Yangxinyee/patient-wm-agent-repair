The second additional batch also passed, with a 0.00373-nat NLL increase. Both checks support retaining the edit. I will now aggregate their exact paired results, verify the weights are unchanged, and recover the remaining rollout batches and causal-estimation notes. Unexpected tensor differences or conflicting evidence would challenge the current choice.

```python
from pathlib import Path
import inspect, re
import numpy as np
import pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

def numpy291(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

print("=== EDIT INTEGRITY ===")
print("Edited tensors:", edited())
cp291 = globals().get("amplitude093_checkpoint")
if isinstance(cp291, dict):
    print("Matches amplitude093:",
          {n: np.array_equal(numpy291(load(n)), numpy291(cp291[n]))
           for n in edited() if n in cp291})

fresh_pairs291 = []
for k in (47, 48):
    original = pd.read_csv(f"/work/fwd_{k}/factual_original.csv")
    current = pd.read_csv(f"/work/fwd_{k}/factual_edited.csv")
    print(f"Forward {k} columns:", original.columns.tolist())
    assert len(original) == len(current)
    for c in ("stay_id", "hour", "event"):
        if c in original:
            assert original[c].equals(current[c]), (k, c)
    nc = [c for c in original.columns if "nll" in c.lower()]
    assert len(nc) == 1, nc
    pair = original[["stay_id", "event"]].copy()
    pair["original_nll"] = original[nc[0]]
    pair["edited_nll"] = current[nc[0]]
    pair["delta"] = pair.edited_nll - pair.original_nll
    fresh_pairs291.append(pair)

fresh_validation291 = pd.concat(fresh_pairs291, ignore_index=True)
cluster291 = fresh_validation291.groupby("stay_id").delta.agg(["sum", "size"])
rng291 = np.random.default_rng(291)
ix291 = rng291.integers(0, len(cluster291), size=(5000, len(cluster291)))
boot291 = (cluster291["sum"].to_numpy()[ix291].sum(axis=1) /
           cluster291["size"].to_numpy()[ix291].sum(axis=1))
print("=== ADDITIONAL 200-STAY FACTUAL CHECK ===")
print("Readings:", len(fresh_validation291))
print("Original NLL:", fresh_validation291.original_nll.mean())
print("Edited NLL:", fresh_validation291.edited_nll.mean())
print("Paired increase:", fresh_validation291.delta.mean())
print("Stay-bootstrap 95% interval:", np.quantile(boot291, [.025, .975]))
print(fresh_validation291.groupby("event").delta.agg(["size", "mean"]))

final_tuning_ids291 = sorted(
    pd.read_csv("/data/tuning.csv", usecols=["stay_id"]).stay_id.unique().tolist())
assert len(final_tuning_ids291) == 400
print("=== REMAINING ROLLOUT BATCHES ===")
print("MIDDLE A:", final_tuning_ids291[100:200])
print("MIDDLE B:", final_tuning_ids291[200:300])

notes291 = Path("/work/notes.md").read_text()
sections291 = re.split(r"(?m)(?=^## )", notes291)
relevant291 = [s for s in sections291
               if re.search(r"causal|identif|target|impulse|drug effect",
                            s.splitlines()[0] if s.splitlines() else "", re.I)]
print("=== CAUSAL EVIDENCE HEADINGS ===")
for s in relevant291:
    print(s.splitlines()[0])
print("=== EARLY IDENTIFICATION EVIDENCE ===")
for s in relevant291[:7]:
    print(s[:1800])

f291 = globals().get("matched_causal_targets163")
if callable(f291):
    print("=== TARGET HELPER ===")
    print("Parameters:", [(p.name, p.kind.name,
                           "required" if p.default is inspect.Parameter.empty
                           else type(p.default).__name__)
                          for p in inspect.signature(f291).parameters.values()])
    print("Referenced names:", f291.__code__.co_names)
    print("Local names:", f291.__code__.co_varnames)
    for name in f291.__code__.co_names:
        if name in globals() and name not in ("np", "pd"):
            obj = globals()[name]
            print(name, type(obj).__name__, getattr(obj, "shape", None))

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Additional factual validation; Python 291, forwards 48/50\n"
        "- Current edit unchanged, hash reported by GPU: 255e915162.\n"
        f"- Prespecified additional 200 training stays: {len(fresh_validation291)} "
        f"readings, paired NLL increase {fresh_validation291.delta.mean():.8f}.\n"
        f"- Stay-bootstrap 95% interval: {np.quantile(boot291, [.025, .975]).tolist()}.\n"
        "- Full400 tuning increase remains +0.00336585. Both checks pass; "
        "unseen-stay compliance is not guaranteed.\n"
        "- Two remaining forward requests reserved for middle200 tuning rollouts.\n")
```

The small-signal route reduced held-out slot error to 0.32, while carrying both innovations with negligible inactive leakage. That is better than the large-anchor route, but it does not establish factual safety. I’ll install this zero-response candidate for a GPU check; an NLL increase above 0.005 would reject it.

```python
import pandas as pd

for name223 in list(edited()):
    drop_edit(name223)
for name223, value223 in route222.items():
    save_edit(name223, np.asarray(value223, dtype=np.float32))

stays223 = pd.read_csv("/data/tuning.csv", usecols=["stay_id"])["stay_id"].unique()
stays223 = sorted(int(x) for x in stays223)[:100]
print("FACTUAL TEST STAYS:", stays223)
print("INSTALLED:", edited())
print("Reserved response-column norms:",
      np.linalg.norm(load("core.prelude.0.attn.out.weight")[:, hs222:hs222+2],
                     axis=0))
with open("/work/notes.md", "a") as f:
    f.write("\n## Python 223 — install zero-response small route for factual validation\n"
            "- Installed small_route222 (11 tensors); reserved innovation-response columns are exactly zero.\n"
            "- Held reconstruction improved, but this is not factual certification.\n"
            "- Next: matched original/edited GPU factual check on first 100 tuning stays; reject if dNLL exceeds .005.\n")
```
Earlier edits produced larger insulin shifts, but often exaggerated halving, so stronger response alone is not enough. I’ll screen lower insulin gates with the recovered factual scorer and review how the norepinephrine response direction was chosen. A candidate with factual headroom can then be tested in the actual rollout.

```python
import numpy as np, torch
from pathlib import Path

def brief264(v):
    if isinstance(v, dict):
        return {str(k): brief264(x) for k,x in v.items()
                if not isinstance(x, (np.ndarray, torch.Tensor))
                or np.size(a262(x)) <= 16}
    if isinstance(v, (np.ndarray, torch.Tensor)):
        x = a262(v)
        return {"shape":tuple(x.shape), "mean":float(np.nanmean(x))}
    if isinstance(v, (tuple,list)):
        return [brief264(x) for x in v[:5]]
    if isinstance(v, (np.floating,np.integer)):
        return v.item()
    return v

saved_globals264 = (rW, rp_linear, repair_bank)
screen264 = {}
try:
    original264 = {n: like258(a262(load_original(n)),base257[n]) for n in NAMES}
    original_score264 = factual_eval194(original264)
    print("ORIGINAL CPU", brief264(original_score264))
    for key in ((1.0,22.0),(1.25,24.0),(1.5,24.0),
                (1.5,26.0),(1.75,26.0),(2.0,28.0)):
        c = candidate258(*key)
        score = factual_eval194(c)
        screen264[key] = {"weights":c, "score":score}
        print("CANDIDATE",key,brief264(score))
finally:
    rW, rp_linear, repair_bank = saved_globals264

print("\nNE RESPONSE CONSTRUCTION")
nt264 = Path("/work/notes.md").read_text()
ss264 = nt264.find("## Run 201")
print(nt264[ss264:ss264+2900] if ss264 >= 0 else "section not found")

print("\nINSULIN OUTPUT PAIR")
wo264 = a262(load("z_post.2.weight"))[:,[96,184]]
print("norms",np.linalg.norm(wo264,axis=0),
      "cosine",float(wo264[:,0]@wo264[:,1] /
                     np.prod(np.linalg.norm(wo264,axis=0))))
with open("/work/notes.md","a") as f:
    f.write("\n## Run 264\nUNSAVED lower-gate insulin factual screen: "
            +repr({k:brief264(v["score"]) for k,v in screen264.items()})
            +". CPU globals restored; saved weights unchanged. "
             "Reviewing recurrent NE response construction.\n")
```

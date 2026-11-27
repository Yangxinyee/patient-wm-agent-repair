The adjusted estimates suggest both drugs have effects: insulin lowers glucose, while norepinephrine raises MAP, with responses lasting several hours. The original rollouts are much weaker. I’ll now verify the attention calculation locally so compensation can be tested before spending another GPU pass.

```python
import numpy as np
from pathlib import Path

r244_attn = "core.prelude.0.attn."
print("ATTENTION TENSORS", [(n, SHAPES[n]) for n in NAMES if n.startswith(r244_attn)])

def r244_attention_features(nt, weights):
    nt = np.asarray(nt, dtype=np.float64)
    def get(s, original=True):
        n = r244_attn + s
        if n in weights:
            return np.asarray(weights[n], dtype=np.float64)
        return np.asarray(load_original(n), dtype=np.float64)
    qw, kvw = get("q.weight"), get("kv.weight")
    qb = get("q.bias") if r244_attn+"q.bias" in NAMES else np.zeros(192)
    kvb = get("kv.bias") if r244_attn+"kv.bias" in NAMES else np.zeros(384)
    q = (nt[:, :4] @ qw.T + qb).reshape(-1,4,4,48).transpose(0,2,1,3)
    kv = (nt @ kvw.T + kvb).reshape(-1,7,2,4,48)
    k = kv[:,:,0].transpose(0,2,1,3)
    v = kv[:,:,1].transpose(0,2,1,3)
    logits = np.einsum("bhqd,bhkd->bhqk", q, k) / np.sqrt(48)
    logits -= logits.max(axis=-1, keepdims=True)
    a = np.exp(logits)
    a /= a.sum(axis=-1, keepdims=True)
    f = np.einsum("bhqk,bhkd->bhqd", a, v)
    return f.transpose(0,2,1,3).reshape(-1,192), a

r244_origfeat, r244_origa = r244_attention_features(relay206["norm_tokens"], {})
r244_hf = np.asarray(relay206["head_features"])
r244_target = np.asarray(relay206["target"])
r244_outw = np.asarray(load_original(r244_attn+"out.weight"))
r244_outb = (np.asarray(load_original(r244_attn+"out.bias"))
             if r244_attn+"out.bias" in NAMES else np.zeros(192))
print("feature reconstruction RMSE", np.sqrt(np.mean((r244_origfeat-r244_hf)**2)))
print("target reconstruction RMSE", np.sqrt(np.mean((r244_origfeat @ r244_outw.T+r244_outb-r244_target)**2)))

print("\nRESERVED HEAD")
r244_q = np.asarray(safe236[r244_attn+"q.weight"])
print("q block norms", np.linalg.norm(r244_q.reshape(4,48,192), axis=(1,2)).round(5))
for n in ("z_in.weight", "z_post.2.weight", "z_post.2.bias"):
    w = np.asarray(safe236[n])
    o = np.asarray(load_original(n))
    d = w-o
    print(n, "difference norm", round(float(np.linalg.norm(d)),4))
    if n.startswith("z_post"):
        print("changed output rows", np.where(np.linalg.norm(d,axis=1)>1e-5)[0].tolist()
              if d.ndim==2 else np.where(np.abs(d)>1e-5)[0].tolist())

print("marker latent coordinates", np.where(np.any(np.abs(T240[:32])>1e-9,axis=1))[0].tolist())
print("relay fit metrics", {k:v["metrics"] for k,v in relay206["fits"].items()})

r244_safefeat, r244_safea = r244_attention_features(nt240, safe236)
r244_head = int(np.argmin(np.linalg.norm(r244_q.reshape(4,48,192),axis=(1,2))))
r244_other = np.setdiff1d(np.arange(192), np.arange(r244_head*48,(r244_head+1)*48))
r244_selected = np.einsum("bqk,bkd->bqd", selector240[16.0]["weights"], nt240).reshape(-1,192)
r244_F = np.column_stack((r244_safefeat[:,r244_other], r244_selected))
r244_M = r244_selected @ V240
print("candidate feature shape",r244_F.shape,"marker RMS",np.sqrt(np.mean(r244_M**2,axis=0)).round(5))
print("recent helper names", ", ".join(
    n for n,x in list(globals().items())
    if callable(x) and hasattr(x,"__code__") and any(s in n for s in ("221","226","227","posterior","step"))
)[:1000])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 244\nReconstructed prelude attention features against cached originals and prepared selective-head compensation features while preserving safe236 latent weights. No weights changed. Counts: 244 Python runs, 23 forwards.\n")
```
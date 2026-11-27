The original prelude activations are also large, so scale alone does not explain the failures. The more important finding is that the cached changed-dose latent tokens differ from the current helper. I’ll reconstruct the full posterior used by the installed weights. A close match to the factual posterior is required before trusting new contrast fits.

```python
import torch.nn.functional as F

def affine261(x, prefix):
    return F.linear(torch.as_tensor(x, dtype=torch.float32),
                    torch.as_tensor(load(prefix+".weight")),
                    torch.as_tensor(load(prefix+".bias")))

def posterior261(ctx, e, u):
    x = torch.cat([torch.as_tensor(ctx,dtype=torch.float32),
                   torch.as_tensor(e,dtype=torch.float32),
                   torch.as_tensor(u,dtype=torch.float32)], dim=-1)
    return affine261(F.gelu(affine261(x,"z_post.0")),"z_post.2")[:,:32]

print("latent206 dependencies:", latent206.__code__.co_names)
print("state_context dependencies:", state_context.__code__.co_names)

ctx_candidates261 = []
for k in sorted(list(globals())):
    if k.startswith("_"):
        continue
    v = globals()[k]
    if hasattr(v,"shape") and tuple(v.shape)==(4213,192) and any(
            s in k.lower() for s in ["ctx","context"]):
        ctx_candidates261.append(k)
print("context candidates:", ctx_candidates261)

ctx_scores261 = []
for k in ctx_candidates261:
    zz = posterior261(globals()[k], tokens259[:,4], u_factual255)
    err = float(torch.sqrt(((zz-z_factual255)**2).mean()))
    ctx_scores261.append((err,k))
print("current posterior reconstruction:", sorted(ctx_scores261))

print("\n=== RELATED CONTEXT / POSTERIOR HELPERS ===")
for k in sorted(list(globals())):
    if k.startswith("_"):
        continue
    v = globals()[k]
    if any(s in k.lower() for s in ["posterior","ctx","context"]) and (
            callable(v) or hasattr(v,"shape")):
        if callable(v):
            print(k, str(inspect.signature(v)), "names", v.__code__.co_names[:12]
                  if hasattr(v,"__code__") else "")
        else:
            print(k, tuple(v.shape))

print("\n=== ORIGINAL-CURRENT FACTUAL DIFFERENCES ===")
for j in [0,2,3]:
    print("part", j, "RMS",
          float(torch.sqrt(((parts259[j]-oparts259[j])**2).mean())))
with open("/work/notes.md","a") as f:
    f.write("\nPython run 261; forwards 20. Original factual raw-state scale is also large, so absolute scale is not itself evidence of a bad edit. Reconstructing the full current posterior rather than relying on partial latent helpers or unverified contrast caches. No weights changed.\n")
```
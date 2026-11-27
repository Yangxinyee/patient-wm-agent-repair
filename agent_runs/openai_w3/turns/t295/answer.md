The inspection stopped on a Python type-checking error; no weights changed. The helper also applies normalization, so its difference from a raw latent token is not yet evidence of stale caches. I’ll fix the inspection and compare full posterior reconstructions directly. An error near numerical precision would validate the inputs for the joint fit.

```python
ctx_candidates262 = []
for k in sorted(list(globals())):
    if k.startswith("_"):
        continue
    v = globals()[k]
    if isinstance(v,(np.ndarray,torch.Tensor)) and tuple(v.shape)==(4213,192):
        if any(s in k.lower() for s in ["ctx","context"]) or k in ["C154","C72","c72"]:
            ctx_candidates262.append(k)
print("context candidates:", ctx_candidates262)

ctx_scores262 = []
for k in ctx_candidates262:
    zz = posterior261(globals()[k], tokens259[:,4], u_factual255)
    err = float(torch.sqrt(((zz-z_factual255)**2).mean()))
    ctx_scores262.append((err,k))
print("current posterior reconstruction:", sorted(ctx_scores262))

for k in ["C154","E154","posterior153","zin154","zb154"]:
    if k in globals():
        v = globals()[k]
        print(k, tuple(v.shape) if isinstance(v,(np.ndarray,torch.Tensor))
              else str(inspect.signature(v)) if callable(v) else type(v).__name__)

print("\n=== FACTUAL PRELUDE DIFFERENCES ===")
for j in [0,2,3]:
    print("part",j,"original-current RMS",
          float(torch.sqrt(((parts259[j]-oparts259[j])**2).mean())))

if ctx_scores262:
    best_error262, best_ctx_name262 = min(ctx_scores262)
    if best_error262 < 1e-3:
        ctx262 = torch.as_tensor(globals()[best_ctx_name262],dtype=torch.float32)
        fresh_tokens262 = {}
        for key,t in r214_tokens.items():
            tt = torch.as_tensor(t).clone()
            zz = posterior261(ctx262,tokens259[:,4],tt[:,5])
            tt[:,6] = affine261(zz,"z_in")
            fresh_tokens262[key] = tt
        print("\nfull changed-dose token reconstruction RMS:",
              [(k,round(float(torch.sqrt(((fresh_tokens262[k]-
                    torch.as_tensor(r214_tokens[k]))**2).mean())),7))
               for k in fresh_tokens262])
    else:
        print("No sufficiently accurate context; no fitting attempted.")

with open("/work/notes.md","a") as f:
    f.write("\nPython run 262; forwards 20. Run261 inspection failed on a module.shape attribute; no edits changed. Partial latent helper includes layer normalization, so raw/helper mismatch was not evidence of stale tokens. Comparing full posterior and changed-dose token reconstructions.\n")
```
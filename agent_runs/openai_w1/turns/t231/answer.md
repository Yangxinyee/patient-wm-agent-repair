The attention measurement supports the transmission hypothesis: slots strongly attend to actions, while latent-token attention is usually negligible. I’ll test a dedicated relay using one uniform-attention head, with a closed-form compensation for its original contribution. I will reject this route if it cannot reproduce factual attention outputs on held-out stays before any weights are installed.

```python
import numpy as np, torch, inspect
from pathlib import Path
torch.set_num_threads(8)

# Use the best reconstructed factual encoders, not older convention probes.
rh206 = np205(rr_prevh4)
re206 = np205(rp_raw_e)
ru206 = np205(rp_raw_u)
rz206 = np205(carrier200_token_original)
rs206 = np205(load_original("core.slots"))
rt206 = np.concatenate((rh206 + rs206[None], re206[:,None],
                        ru206[:,None], rz206[:,None]), axis=1)
rn206 = torch.nn.functional.layer_norm(
    torch.as_tensor(rt206, dtype=torch.float32), (192,),
    torch.as_tensor(np205(load_original("core.prelude.0.norm1.weight"))),
    torch.as_tensor(np205(load_original("core.prelude.0.norm1.bias"))))
rq206 = torch.nn.functional.linear(
    rn206, torch.as_tensor(np205(load_original("core.prelude.0.attn.q.weight"))))
rkv206 = torch.nn.functional.linear(
    rn206, torch.as_tensor(np205(load_original("core.prelude.0.attn.kv.weight"))))
rk206, rv206 = rkv206.chunk(2, -1)
rq206 = rq206.reshape(-1,7,4,48).transpose(1,2)
rk206 = rk206.reshape(-1,7,4,48).transpose(1,2)
rv206 = rv206.reshape(-1,7,4,48).transpose(1,2)
ra206 = torch.softmax(rq206 @ rk206.transpose(-1,-2)/np.sqrt(48), -1)
rf206 = (ra206[:,:,:4] @ rv206).transpose(1,2).reshape(-1,192).numpy()
ro206 = np205(load_original("core.prelude.0.attn.out.weight"))
ry206 = rf206 @ ro206.T
rp206 = np.repeat(rn206.mean(1).numpy(), 4, axis=0)
train206 = np.repeat(np.asarray(rp_train, dtype=bool), 4)
hold206 = np.repeat(np.asarray(rp_hold, dtype=bool), 4)
assert train206.shape == hold206.shape == (len(ry206),)
assert not np.any(train206 & hold206)

relay206 = dict(norm_tokens=rn206.numpy(), head_features=rf206, target=ry206,
                pool=rp206, train=train206, hold=hold206, fits={})
print("Factual RMS:", np.sqrt(np.mean(ry206**2)),
      "train/hold slot samples:", train206.sum(), hold206.sum())
for head206 in range(4):
    other206 = np.r_[np.arange(head206*48), np.arange((head206+1)*48,192)]
    x206 = np.concatenate((rf206[:,other206], rp206), axis=1).astype(np.float64)
    xt206, yt206 = x206[train206], ry206[train206].astype(np.float64)
    gram206 = xt206.T @ xt206 / len(xt206)
    ridge206 = 1e-5 * max(np.trace(gram206)/len(gram206), 1e-8)
    coef206 = np.linalg.solve(gram206 + ridge206*np.eye(336),
                             xt206.T @ yt206 / len(xt206))
    # The replacement head has 46 factual features and two carrier features.
    left206, sval206, right206 = np.linalg.svd(coef206[144:], full_matrices=False)
    low206 = (left206[:,:46]*sval206[:46]) @ right206[:46]
    pred206 = x206[:,:144] @ coef206[:144] + x206[:,144:] @ low206
    err206 = pred206 - ry206
    metrics206 = {
        "train_RMSE": float(np.sqrt(np.mean(err206[train206]**2))),
        "hold_RMSE": float(np.sqrt(np.mean(err206[hold206]**2))),
        "hold_bundle_error_p95": float(np.quantile(
            np.sqrt(np.mean(err206[hold206]**2,axis=1)), .95)),
        "original_head_contribution_RMS": float(np.sqrt(np.mean(
            (rf206[:,head206*48:(head206+1)*48] @
             ro206[:,head206*48:(head206+1)*48].T)**2)))
    }
    relay206["fits"][head206] = dict(other=other206, coef=coef206,
                                   left=left206, singular=sval206,
                                   right=right206, metrics=metrics206)
    print("replace head", head206, metrics206)

print("\n=== DOSE CARRIER CONSTRUCTION ===")
for drug206, spec206 in rp_gate_fits.items():
    print(drug206, "tuple length", len(spec206),
          [(tuple(v.shape) if hasattr(v,"shape") else
            list(v.keys()) if isinstance(v,dict) else type(v).__name__)
           for v in spec206])
for name206 in sorted(list(globals())):
    if name206.startswith("lc_"):
        obj206 = globals()[name206]
        if isinstance(obj206, (np.ndarray,torch.Tensor)):
            print(name206, tuple(obj206.shape))
        elif isinstance(obj206,dict):
            print(name206, "keys", list(obj206.keys())[:8])
try:
    print(inspect.getsource(rp_install)[:2300])
except Exception:
    print("rp_install source unavailable; no disassembly printed.")

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 206\nPrelude attention overwhelmingly reads action tokens and usually ignores latent tokens. Tested closed-form factual compensation for a uniform single-head carrier relay, with two reserved carrier features and stay-held-out local validation. No weights installed. Counts: 206 Python runs, 18 forwards.\n")
```
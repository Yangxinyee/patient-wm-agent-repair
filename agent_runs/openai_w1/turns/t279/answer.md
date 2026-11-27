Small finite perturbations have the intended signs, though larger ones become nonlinear. I’ll install a reversible, moderate-gain candidate using the lower-norm insulin direction and the more channel-specific norepinephrine direction. A paired factual test must pass before I treat its rollout changes as useful.

```python
import numpy as np
from pathlib import Path

r247_R = r245_fits[1.0]["R"]
r247_B = np.column_stack((-0.5*_Bmean227[:,0], 0.4*B221[:,1]))
r247_C = r245_Cbase + r247_R @ r247_B.T
r247_U, r247_s, r247_Vt = np.linalg.svd(r247_C[144:],full_matrices=False)
r247_rank = 48
r247_root = np.sqrt(r247_s[:r247_rank])
r247_value = (r247_U[:,:r247_rank]*r247_root).T
r247_output = (r247_root[:,None]*r247_Vt[:r247_rank]).T
r247_Cactual = r247_C.copy()
r247_Cactual[144:] = r247_value.T @ r247_output.T
r247_pred = r245_F @ r247_Cactual
print("held factual attention RMSE",
      np.sqrt(np.mean((r247_pred[r245_hold]-r244_target[r245_hold])**2)))
print("rank truncation RMS",np.sqrt(np.mean((r245_F@(r247_Cactual-r247_C))**2)))

for key in (("insulin",2.0),("insulin",0.5),
            ("norepinephrine",2.0),("norepinephrine",0.5)):
    item = r245_deltas[key]
    df = item["delta"][item["hold"]].astype(np.float64)
    added = (df@r247_R)@r247_B.T
    mean_added = added.mean(axis=0)
    print(key,"added attention mean norm",round(float(np.linalg.norm(mean_added)),4),
          "local reading shifts",np.asarray(finite_readout221(mean_added.astype(np.float32))).round(4))

r247_candidate = {n:np.asarray(w,dtype=np.float32).copy() for n,w in safe236.items()}
r247_q = r247_candidate[r244_attn+"q.weight"]
r247_kv = r247_candidate[r244_attn+"kv.weight"]
r247_ow = r247_candidate[r244_attn+"out.weight"]
r247_rows = np.arange(r244_head*48,(r244_head+1)*48)
r247_q[r247_rows] = 0
r247_q[r247_rows[0]] = qvec240.astype(np.float32)
r247_kv[r247_rows] = 0
r247_kv[r247_rows[0]] = (16*np.sqrt(48)*kvec240).astype(np.float32)
r247_kv[192+r247_rows] = r247_value.astype(np.float32)
r247_ow[:,r244_other] = r247_Cactual[:144].T.astype(np.float32)
r247_ow[:,r247_rows] = r247_output.astype(np.float32)

for n,w in r247_candidate.items():
    save_edit(n,w)
r247_checkfeat,_ = r244_attention_features(nt240,r247_candidate)
r247_check = r247_checkfeat @ r247_candidate[r244_attn+"out.weight"].T
print("installed attention reconstruction RMSE",
      np.sqrt(np.mean((r247_check-r247_pred)**2)))
print("edited tensors",edited())

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 247\nInstalled reversible selective latent-attention candidate without replacing safe236 z_in: dose-invariant compensation weight1; insulin output -.5*Bmean, NE +.4*B221; rank48 attention factorization. Local finite responses only guide signs, not rollout efficacy. Paired GPU factual validation next. Counts: 247 Python runs, 23 forwards.\n")
```
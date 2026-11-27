Factual compensation preserves most dose contrasts, and the lower-norm insulin direction behaves better. Norepinephrine still saturates, so efficacy is uncertain. I’ll install a reversible candidate using these directions and refit the factual relay compensation. Paired GPU checks will confirm or reject both factual safety and rollout improvement.

```python
import numpy as np, torch, pandas as pd
import torch.nn.functional as F

def _attention_features228(nt, qw, kvw):
    x = torch.as_tensor(np.asarray(nt),dtype=torch.float32)
    q = F.linear(x,torch.as_tensor(qw,dtype=torch.float32))
    kv = F.linear(x,torch.as_tensor(kvw,dtype=torch.float32))
    for suffix, dst in [("q.bias","q"),("kv.bias","kv")]:
        name = "core.prelude.0.attn."+suffix
        if name in NAMES:
            b = torch.as_tensor(load(name),dtype=torch.float32)
            if dst=="q": q=q+b
            else: kv=kv+b
    n,t,_=x.shape
    q=q.reshape(n,t,4,48).permute(0,2,1,3)
    k,v=kv.chunk(2,dim=-1)
    k=k.reshape(n,t,4,48).permute(0,2,1,3)
    v=v.reshape(n,t,4,48).permute(0,2,1,3)
    a=torch.softmax((q@k.transpose(-1,-2))/np.sqrt(48),dim=-1)
    return (a@v).permute(0,2,1,3)[:,:4].reshape(-1,192).numpy().astype(np.float64)

candidate228_baseline = {n: np.asarray(load(n)).copy() for n in edited()}
candidate228_carrier_names = [
    "z_post.0.weight","z_post.0.bias","z_post.2.weight","z_post.2.bias",
    "z_in.weight","z_in.bias"]
candidate228_q = np.asarray(load("core.prelude.0.attn.q.weight")).copy()
candidate228_kv = np.asarray(load("core.prelude.0.attn.kv.weight")).copy()
candidate228_out0 = np.asarray(load("core.prelude.0.attn.out.weight")).copy()
candidate228_out = candidate228_out0.copy()
candidate228_rows = np.arange(96,144)
candidate228_reserved = candidate228_rows[-2:]
candidate228_kv[192+candidate228_reserved] = V215.astype(np.float32)
candidate228_gain = np.array([-2.0,3.3])
candidate228_fixed = (_Bmean227*candidate228_gain).astype(np.float32)
candidate228_out[:,candidate228_reserved] = candidate228_fixed

_nt228 = np.asarray(nt215)
_expected_z228 = norm215(zf215 @ zin215.T + lc_zbias_base)
print("Latent-token reconstruction RMSE:",
      float(np.sqrt(np.mean((_expected_z228-_nt228[:,6])**2))))
_basefeat228 = _attention_features228(relay206["norm_tokens"],candidate228_q,
                                      np.asarray(load("core.prelude.0.attn.kv.weight")))
_target228 = np.asarray(relay206["target"],dtype=np.float64)
print("Current relay attention RMSE:",
      float(np.sqrt(np.mean((_basefeat228@candidate228_out0.T-_target228)**2))))

_feat228 = _attention_features228(_nt228,candidate228_q,candidate228_kv)
_n228 = len(_nt228)
_otherrows228 = np.setdiff1d(np.arange(192),candidate228_rows)
_otherout228 = (_feat228[:,_otherrows228] @
                candidate228_out0[:,_otherrows228].T).reshape(_n228,4,192).mean(axis=1)
_headfeat228 = _feat228[:,candidate228_rows].reshape(_n228,4,48).mean(axis=1)
_q228 = _headfeat228[:,:46]
_f228 = _headfeat228[:,46:]
_y228 = (_target228.reshape(_n228,4,192).mean(axis=1)
         -_otherout228-_f228@candidate228_fixed.T)
_gram228 = _q228[_train227].T@_q228[_train227]
_lam228 = 1e-5*np.trace(_gram228)/46
_coef228 = np.linalg.solve(
    _gram228+_lam228*np.eye(46),
    _q228[_train227].T@_y228+
    _lam228*candidate228_out0[:,candidate228_rows[:46]].T)
candidate228_out[:,candidate228_rows[:46]] = _coef228.T.astype(np.float32)
_err228 = _feat228@candidate228_out.T-_target228
print("Candidate attention RMSE train/hold:",
      float(np.sqrt(np.mean(_err228[np.asarray(relay206["train"])]**2))),
      float(np.sqrt(np.mean(_err228[np.asarray(relay206["hold"])]**2))))

candidate228_weights = {
    "core.prelude.0.attn.q.weight":candidate228_q,
    "core.prelude.0.attn.kv.weight":candidate228_kv,
    "core.prelude.0.attn.out.weight":candidate228_out,
    "z_post.0.weight":carrier200_w0,
    "z_post.0.bias":carrier200_b0,
    "z_post.2.weight":carrier200_w2,
    "z_post.2.bias":carrier200_b2,
    "z_in.weight":zin215,
    "z_in.bias":lc_zbias_base}
assert np.sqrt(np.mean((_expected_z228-_nt228[:,6])**2)) < 1e-3
for _name228,_weight228 in candidate228_weights.items():
    save_edit(_name228,torch.as_tensor(np.asarray(_weight228),dtype=torch.float32))
print("Installed edits:",edited())

_tuning228=pd.read_csv("/data/tuning.csv")
print("Tuning columns:",list(_tuning228.columns))
_staycol228="stay_id" if "stay_id" in _tuning228.columns else "stay"
candidate228_stays=list(map(int,_tuning228[_staycol228].drop_duplicates().iloc[:100]))
print("GPU STAYS:",candidate228_stays)

with open("/work/notes.md","a") as _f228:
    _f228.write("\n## Run 228\nInstalled reversible marker+relay candidate: insulin attention gain -2 using lower-norm mean-response direction; NE +3.3. Refit 46 factual relay features with closed-form ridge while fixing two marker output channels. Candidate must pass paired GPU factual and rollout tests; no efficacy claim yet.\n")
```
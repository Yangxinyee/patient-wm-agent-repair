The newer cache still has a meaningful reconstruction error, so it is not yet a safe basis for the edit. I’ll compare the preserved context variants and rebuild the action encodings. Anchoring dose contrasts to the best cached factual input will prevent reconstruction errors from masquerading as treatment effects.

```python
import numpy as np, torch
torch.set_num_threads(8)

_rp_valid = np.asarray(r167_same, dtype=bool) & np.asarray(r168_valid, dtype=bool)
_rp_truth = torch.as_tensor(r168_zactual)
_rp_candidates = []
for _ck,_ctx in r167_contexts.items():
    for _ek,_e in (("ds_e",torch.as_tensor(ds_in[:,192:384])),("rr_e",rr_e4)):
        for _uk,_u in (("ds_u",torch.as_tensor(ds_u)),("rr_u",rr_u4)):
            _inp=torch.cat([torch.as_tensor(_ctx),_e,_u],dim=1).float()
            _hid=torch.nn.functional.gelu(_inp@ds_w0.T+ds_b0)
            _zp=(_hid@ds_w2.T+ds_b2)[:,:32]
            _err=(_zp-_rp_truth).numpy()
            _rms=float(np.sqrt(np.mean(_err[_rp_valid]**2)))
            _rp_candidates.append((_rms,_ck,_ek,_uk,_inp,_hid))
_rp_candidates.sort(key=lambda x:x[0])
print("=== BEST INPUT COMBINATIONS ===")
for _r in _rp_candidates[:6]:
    print(_r[:4])
_,rp_context_choice,rp_e_choice,rp_u_choice,rp_input,rp_hidden=_rp_candidates[0]
rp_ctx=rp_input[:,:192].numpy()
rp_valid=_rp_valid
print("best valid/active RMSE:",
      float(np.sqrt(np.mean(((_rp_candidates[0][5]@ds_w2.T+ds_b2)[:,:32].numpy()-r168_zactual)[rp_valid]**2))),
      [float(np.sqrt(np.mean(((_rp_candidates[0][5]@ds_w2.T+ds_b2)[:,:32].numpy()-r168_zactual)[rp_valid&(ds_rates[:,j]>0)]**2))) for j in range(2)])

rp_meta=[]; rp_owner_list=[]
for _i in range(len(ds_rates)):
    _key=(int(ds_s["stay_id"][_i]),int(round(float(ds_s["hour"][_i])*60)))
    for _a in rr_bundle_map4.get(_key,[]):
        rp_meta.append(dict(_a)); rp_owner_list.append(_i)
rp_owner=torch.tensor(rp_owner_list,dtype=torch.long)
rp_atoms=torch.stack([rr_atom(**_a) for _a in rp_meta])
rp_drug=np.array([a["code"] in ("insulin","norepinephrine") for a in rp_meta])
rp_end=np.array([bool(a.get("end",False)) for a in rp_meta])

def rp_encode(atoms,mask,prefix,empty_zero=False):
    mask=torch.as_tensor(mask,dtype=torch.bool)
    owner=rp_owner[mask]
    count=torch.bincount(owner,minlength=len(ds_rates)).float()
    summed=torch.zeros(len(ds_rates),192)
    summed.index_add_(0,owner,rr_mlp(atoms[mask],prefix+".mlp1"))
    v=summed/count.clamp_min(1).sqrt()[:,None]+rr_linear(torch.log1p(count)[:,None],prefix+".size")
    enc=rr_norm(rr_mlp(v,prefix+".mlp2"),prefix+".norm")
    if empty_zero: enc[count==0]=0
    return enc

rp_encodings={}
for _mode in ("all_drug_actions","drug_END_observation","drug_END_ignored"):
    _am=rp_drug if _mode=="all_drug_actions" else (rp_drug&~rp_end)
    _om=~rp_drug if _mode!="drug_END_observation" else (~rp_drug|rp_end)
    _u=rp_encode(rp_atoms,_am,"encode_action",True)
    for _zero in (False,True):
        _e=rp_encode(rp_atoms,_om,"encode",_zero)
        rp_encodings[(_mode,_zero)]=(_e,_u,_om,_am)
print("=== RAW ENCODER MATCH TO BEST CACHED INPUT ===")
_rp_enc_scores=[]
for _key,(_e,_u,_om,_am) in rp_encodings.items():
    _er=float(torch.sqrt(torch.mean((_e-rp_input[:,192:384])**2)))
    _ur=float(torch.sqrt(torch.mean((_u-rp_input[:,384:])**2)))
    _rp_enc_scores.append((_er+_ur,_key,_er,_ur))
for _r in sorted(_rp_enc_scores):
    print(_r)
rp_encoder_choice=sorted(_rp_enc_scores)[0][1]
rp_raw_e,rp_raw_u,rp_obs_mask,rp_action_mask=rp_encodings[rp_encoder_choice]
print("chosen:",rp_context_choice,rp_e_choice,rp_u_choice,rp_encoder_choice,
      "atoms",len(rp_meta),"valid rows",int(rp_valid.sum()))
with open("/work/notes.md","a") as _f:
    _f.write("\n## Run 173\nCompared preserved fresh contexts and encoder conventions. Dose contrasts will be anchored to the best factual input, so raw-encoder mismatch cannot create a spurious baseline correction. No new edit yet.\n")
```
Layer normalization matches exactly, but the check stopped because the context helper returns a tuple. I also found that admission bundles can contain readings, so their memory vectors must combine both. I’ll correct those two issues and rerun the transition comparison; a small residual error would validate the reconstruction.

```python
probe_context = context_from_saved(np.array([int(wm_drug_rows[0])]), s4)
print("Context return:",
      [(type(v).__name__, tuple(v.shape) if hasattr(v,"shape") else None)
       for v in probe_context] if isinstance(probe_context,tuple)
      else (type(probe_context).__name__, tuple(probe_context.shape)))

visit_z_errors = []
for j in range(len(s4["x"])):
    kk = (int(s4["stay_id"][j]), int(round(s4["hour"][j]*60)))
    gg = groups4.get(kk)
    if gg is None:
        continue
    aa = []
    has_visit = False
    for rr in gg.itertuples(index=False):
        if rr.event in measurement_unit_local:
            aa.append(measurement_atom_local(rr.event,[rr.value],
                                             measurement_unit_local[rr.event])[0])
        elif rr.event in ["hospital_admission","icu_admission"]:
            has_visit = True
            aa.append(W0["embed.code.weight"][code_id[rr.event]]
                      + W0["embed.kind.weight"][0] + W0["embed.ekind.weight"][3]
                      + W0["embed.kq.weight"][0] + W0["embed.numeric.bias"])
    wm_e4[j] = wm_set(torch.stack(aa) if aa else torch.empty(0,192), "encode")
    if has_visit and j>0 and s4["stay_id"][j]==s4["stay_id"][j-1] and len(visit_z_errors)<20:
        cv = context_from_saved(np.array([j]),s4)
        if isinstance(cv,tuple):
            cv = cv[0]
        zp = wm_mlp(torch.cat([cv,wm_e4[j:j+1],empty_action_local],-1),"z_post")[:,:32]
        visit_z_errors.append(float((zp-torch.from_numpy(s4["x"][j:j+1,960:])).square().mean().sqrt()))
print("Admission latent RMSE:",np.mean(visit_z_errors))

wm_kv_layout = "global"
wm_memory_mode = "raw"
def wm_attention(x,prefix,memory=None,agevec=None,age_mode="input",weights=None):
    source = x if memory is None else memory
    q = wm_lin(x,prefix+".q",weights)
    if memory is not None:
        norm_prefix = prefix.rsplit(".",1)[0]+".norm2"
        if wm_memory_mode=="norm_then_age":
            source = wm_norm(source,norm_prefix,weights)
        if agevec is not None and age_mode=="input":
            source = source+agevec
        if wm_memory_mode=="age_then_norm":
            source = wm_norm(source,norm_prefix,weights)
        elif wm_memory_mode=="age_then_plainnorm":
            source = F.layer_norm(source,(192,))
    batch,nt,_ = q.shape
    ns = source.shape[1]
    kv = wm_lin(source,prefix+".kv",weights)
    if wm_kv_layout=="global":
        k,v = kv.chunk(2,-1)
        k = k.reshape(batch,ns,4,48).transpose(1,2)
        v = v.reshape(batch,ns,4,48).transpose(1,2)
    else:
        kv = kv.reshape(batch,ns,4,2,48)
        k,v = kv[:,:,:,0,:].transpose(1,2),kv[:,:,:,1,:].transpose(1,2)
    if agevec is not None and age_mode=="key":
        k = k+agevec.reshape(batch,ns,4,48).transpose(1,2)
    q = q.reshape(batch,nt,4,48).transpose(1,2)
    prob = torch.softmax(q@k.transpose(-2,-1)/np.sqrt(48),-1)
    result = (prob@v).transpose(1,2).reshape(batch,nt,192)
    return wm_lin(result,prefix+".out",weights)

wm_transition_trials2 = []
for wm_kv_layout in ["global","perhead"]:
    for wm_memory_mode in ["raw","norm_then_age","age_then_norm","age_then_plainnorm"]:
        for inc in [False,True]:
            for am in ["input","key"]:
                errors,outerrors = [],[]
                for j in wm_drug_rows[:3]:
                    pp = wm_step_saved(int(j),inc,am,"seconds")
                    yy = torch.from_numpy(s4["x"][j,192:960]).reshape(1,4,192)
                    errors.append(float((pp-yy).square().mean().sqrt()))
                    outerrors.append(float((wm_norm(pp.mean(1),"core.out_norm")
                                            -torch.from_numpy(s4["x"][j:j+1,:192])).square().mean().sqrt()))
                wm_transition_trials2.append((np.mean(errors),np.mean(outerrors),
                                              wm_kv_layout,wm_memory_mode,inc,am))
print("Best recurrence conventions:",sorted(wm_transition_trials2)[:10])
j = int(wm_drug_rows[0])
print("Target slot std:",float(torch.from_numpy(s4["x"][j,192:960]).std()))
```

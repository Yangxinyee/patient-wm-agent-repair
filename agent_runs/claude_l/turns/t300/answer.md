The rollout replica in /work/wm2_rollout.py is clear now. Conditions are `(drug_code, factor)` (8 = insulin, 9 = norepinephrine), and it records each path's predicted mean for every hidden reading. The builder `mkW3(Gi, Gn, base)` and all its inputs (`GATE`, `DEC2`, `PH`, `VD_`, `Pk`) still exist. Several earlier dicts (`Wt`, `We`, `Wk`) hold older z_post/z_in candidates. `W_cur` equals my current submitted state (only `z_head.weight` edited).

Next I'll build `W1 = mkW3(1, 1, W_cur)` and run the full offline evaluation:
- the four rollout readouts on the 100 batch-1 stays;
- the open-loop "truth" for the same stay-hours, from the record-based impulse responses;
- the exact teacher-forced value NLL, against `W_cur`.

If the edit works, the shifts should move from about −3.3/+1.8/+0.6/−0.8 toward the truth while NLL stays flat. Then the gains just need scaling. If the shifts don't move, or NLL jumps, the builder or the gating is wrong.

```python
import time
t0=time.time()
def cid(name):
    if name in CODE_ID: return CODE_ID[name]
    for k,v in CODE_ID.items():
        if v==name: return k
CF=cid('glucose_finger'); CA=cid('map_arterial')
print('CF,CA',CF,CA,'IRF_I',np.round(IRF_I,2),'IRF_N',np.round(IRF_N,2))
conds=[None,(8,2.0),(8,0.5),(9,2.0),(9,0.5)]
def drug_recs(B_, code):
    nm=DRUGN[code]; rm,ri=NUMV[(nm,'rate')]['median'],NUMV[(nm,'rate')]['iqr']; o=[]
    for b in B_:
        for a in b[1]:
            if a.get('act') and a['code']==code and a['kind']==0: o.append((b[0], a['num'][4]*ri+rm))
    return o
def evalW(W, stays, npaths=20, seed=1, cl=conds):
    acc={ci:[] for ci in range(1,len(cl))}; tru={ci:[] for ci in range(1,len(cl))}
    for s in stays:
        B_=BUN[s]; rec=rollout(W,B_,cl,npaths,seed)
        for ci,(dc,f) in enumerate(cl[1:],1):
            c=CF if dc==8 else CA; irf=IRF_I if dc==8 else IRF_N
            dr=[(t,r) for t,r in drug_recs(B_,dc) if t>=12.0-1e-9]
            for (H,cc),arr in rec.items():
                if cc!=c or H<13 or H>24: continue
                ds_=[(t,r) for t,r in dr if t<H]
                if not ds_: continue
                acc[ci].append(float((arr[ci]-arr[0]).mean()))
                tru[ci].append(sum(irf[H-int(math.floor(t))-1]*(f-1)*r for t,r in ds_ if 1<=H-int(math.floor(t))<=len(irf)))
    return {ci:(round(float(np.mean(acc[ci])),3),round(float(np.mean(tru[ci])),3),len(acc[ci])) for ci in acc}
W1=mkW3(1.0,1.0,W_cur)
print('W1 diff keys',[k for k in W1 if not torch.equal(torch.as_tensor(W1[k]).double(),torch.as_tensor(W_cur[k]).double())])
stays=list(BUN.keys())
r_cur=evalW(W_cur,stays); print('W_cur (pred,true,n):',r_cur, round(time.time()-t0))
r_1=evalW(W1,stays); print('W1    (pred,true,n):',r_1, round(time.time()-t0))
def tfnll(W, stays):
    L=[]; 
    for s in stays: L+= [x[3] for x in tf_stay_z(W,BUN[s],None)]
    return float(np.mean(L)), len(L)
print('NLL W_cur',tfnll(W_cur,stays),'W1',tfnll(W1,stays), round(time.time()-t0))
with open('/work/notes.md','a') as fh:
    fh.write("\n## Restart 6 (run 273+)\n- No cell history; sources lost. mkW3(Gi,Gn,base) = latest z-channel builder (K=30 PCA copy of u, GATE, DEC2 dose decode, PH pihat, VD_ token dirs).\n- evalW(W,stays) = offline rollout readouts vs IRF truth on BUN (batch1); tfnll(W,stays) exact TF NLL via tf_stay_z.\n")
```
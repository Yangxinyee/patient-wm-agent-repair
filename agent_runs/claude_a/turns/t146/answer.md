The multi-head linear fits overfit badly. They look fine in-sample, but on 100 held-out stays the off-target error grows to ±3–5 rate units and the insulin residual doubles. Head 3 alone also degrades on held-out data (insulin residual 0.40, reading-bundle bias −0.43). A purely linear staging of the rate is not viable.

The workable option is a nonlinear gate in the prelude FFN. Two repurposed GELU units pass the staged rate when head 3 has read a drug token and block it otherwise. This needs two things from head 3's 48-dim attended values:
1. A rate readout that is precise at drug bundles and merely bounded elsewhere, since the gate removes the rest.
2. A gate readout that separates drug-start bundles from all others with a wide margin.

Next I map the trade-off: insulin residual against off-target magnitude for small off-target weights, all on held-out stays. I also fit a ±1 classifier and measure its held-out margin. Workable numbers would be an insulin residual ≈0.07 with off-target values within about ±5, and a classifier whose minimum drug-bundle score clears the maximum non-drug score.

```python
A3tr=Ftr[:,0:0]  # placeholder
def A3_of(ds_,ST_):
    T_,n1_,v_=prelude_parts(P, ST_['hf'], ds_['e'], ds_['u'], ST_['z'])
    return torch.einsum('bqk,bkd->bqd', ST_['aw'][:,3,:4,:], v_[:,:,3,:]).mean(1).numpy()
A3tr=A3_of(ds2,ST2); A3va=A3_of(dsv,STv)
drug_tr=fltr['ins']|fltr['ne']; drug_va=flva['ins']|flva['ne']
for wo_ in [0.0,1e-4,1e-3,1e-2,1e-1]:
    Xa=np.c_[A3tr,np.ones(len(A3tr))]; wt=np.where(drug_tr,1.0,wo_); sw=np.sqrt(wt)
    w_,_,_,_=np.linalg.lstsq(Xa*sw[:,None], rtr*sw, rcond=None)
    pv_=np.c_[A3va,np.ones(len(A3va))]@w_
    off_=~drug_va
    print(f"woff={wo_:7.4f} VALID ins resid {np.std(pv_[flva['ins']]-rva[flva['ins']]):.3f} ne_only {np.std(pv_[flva['ne']&~flva['ins']]):.3f} | off-target abs p50 {np.percentile(np.abs(pv_[off_]),50):.2f} p99 {np.percentile(np.abs(pv_[off_]),99):.2f} max {np.abs(pv_[off_]).max():.2f}")
# gate classifier
yc=np.where(drug_tr,1.0,-1.0); Xa=np.c_[A3tr,np.ones(len(A3tr))]
wc_,_,_,_=np.linalg.lstsq(Xa, yc, rcond=None); sc_tr=Xa@wc_; sc_va=np.c_[A3va,np.ones(len(A3va))]@wc_
print('gate TRAIN drug min', sc_tr[drug_tr].min().round(3), 'nondrug max', sc_tr[~drug_tr].max().round(3))
print('gate VALID drug min', sc_va[drug_va].min().round(3), 'p1', np.percentile(sc_va[drug_va],1).round(3), '| nondrug max', sc_va[~drug_va].max().round(3), 'p99', np.percentile(sc_va[~drug_va],99).round(3))
```
The graft uses two value-head units to apply a gated bin-logit tilt, and its gate currently includes both glucose sensors. I’ll measure the gate by event and construct a record-fitted finger-only separator, without installing it yet. Low cached factual cost and retained synthetic dose leverage would justify a GPU trial; loss of leverage or leakage into other readings would reject it.

```python
print("FACTUAL CACHE SHAPES AND GATE DISTRIBUTIONS")
for split, c in sc_fact.items():
    print(split, {k: tuple(v.shape) for k,v in c.items()
                  if hasattr(v,"shape") and not callable(getattr(v,"shape",None))})
    ev = c["meta"]["event"].to_numpy()
    valid = c["meta"]["has_last"].to_numpy().astype(bool)
    for e in sorted(set(ev)):
        mask = (ev==e)&valid
        if mask.any():
            print(e, "n",int(mask.sum()),
                  "gate quantiles",np.round(np.quantile(c["gate"][:,mask],[.01,.1,.5,.9,.99]),2))
print("\nSYNTHETIC CACHE SHAPES")
for mult,c in sc_syn.items():
    print(mult, {k: tuple(v.shape) for k,v in c.items()
                 if hasattr(v,"shape") and not callable(getattr(v,"shape",None))})
print("\nSC_Q protection")
code_w_now = to_np0(load_original("embed.code.weight"))
dcode_now = code_w_now[6]-code_w_now[10]
print("projected finger-lab norm",np.linalg.norm(sc_Q.T@dcode_now))
print("sc_active summaries", {k:(type(v).__name__,getattr(v,"shape",None)) for k,v in sc_active.items()})
print("\nPARAMETER-FREE FINGER/LAB SEPARATOR PROBES")
Q_gate = np.asarray(sc_Q,dtype=np.float64)
F_gate = np.asarray(sc_F,dtype=np.float64)
mu_gate = F_gate.mean(0)
A_gate = (F_gate-mu_gate)@Q_gate
C_gate = A_gate.T@A_gate/len(A_gate)
d_gate = Q_gate.T@np.asarray(dcode_now,dtype=np.float64)
gate_sep_candidates = {}
for ridge in [0.01,0.1,1.0,10.0]:
    vq = np.linalg.solve(C_gate + ridge*np.eye(C_gate.shape[0]),d_gate)
    v = Q_gate@vq/(d_gate@vq)
    gate_sep_candidates[ridge] = v
    for name,c in [("train",sc_fact["first"]),("tuning",sc_fact["tuning"])]:
        means = np.asarray(c["x"]).mean(0)
        ev=c["meta"]["event"].to_numpy()
        valid=c["meta"]["has_last"].to_numpy().astype(bool)
        base_mean = np.mean(means[(ev=="glucose_finger")&valid],axis=0)
        proj=(means-base_mean)@v
        vals={}
        for e in ["glucose_finger","glucose_lab","map_arterial","map_cuff"]:
            m=(ev==e)&valid
            vals[e]=np.round(np.quantile(proj[m],[.01,.5,.99]),3).tolist()
        print("ridge",ridge,name,vals)
```

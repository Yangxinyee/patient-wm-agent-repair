The final-layer decoder partly recovers insulin contrasts but nearly misses doubled norepinephrine. Its pooled score hides that failure, so I won’t install it. I’ll test a decoder before the posterior’s nonlinear layer, with a drug-presence gate. Better held-out x2 accuracy would show that the usable dose signal survives in the action input.

```python
import numpy as np, torch
from scipy.linalg import solve

rp_cf_input={}
for _drug in ("insulin","norepinephrine"):
    _indices=[i for i,a in enumerate(rp_meta) if a["code"]==_drug]
    for _mult in rp_multipliers:
        _atoms=rp_atoms.clone()
        for _i in _indices:
            _a=dict(rp_meta[_i])
            for _field in ("rate","dose"):
                if np.isfinite(_a.get(_field,np.nan)): _a[_field]*=_mult
            _atoms[_i]=rr_atom(**_a)
        _e=rp_encode(_atoms,rp_obs_mask,"encode",rp_encoder_choice[1])
        _u=rp_encode(_atoms,rp_action_mask,"encode_action",True)
        _inp=rp_input.clone()
        _inp[:,192:384]+=_e-rp_raw_e
        _inp[:,384:]+=_u-rp_raw_u
        rp_cf_input[(_drug,_mult)]=_inp.numpy()

rp_X=rp_input.numpy().astype(np.float64)
rp_xmu=rp_X[rp_train].mean(axis=0)
rp_xsd=np.maximum(rp_X[rp_train].std(axis=0),0.05)
rp_Xa=np.column_stack([(rp_X-rp_xmu)/rp_xsd,np.ones(len(rp_X))])
rp_linear_fits={}
rp_gate_fits={}
print("=== INPUT-LEVEL GATED DOSE DECODERS ===")
for _j,_drug in enumerate(("insulin","norepinephrine")):
    _active=ds_rates[:,_j]>0
    _tr=rp_train&_active
    _ho=rp_hold&_active
    _Dd=[];_td=[]
    for (_d,_mult),_x in rp_cf_input.items():
        _D=np.column_stack([(_x-rp_X)/rp_xsd,np.zeros(len(rp_X))])
        _target=(_mult-1)*ds_rates[:,_j] if _d==_drug else np.zeros(len(rp_X))
        _mask=_tr&rp_cf_masks[_d]
        _Dd.append(_D[_mask]);_td.append(_target[_mask])
    _Dd=np.vstack(_Dd);_td=np.concatenate(_td)
    _Gd=_Dd.T@_Dd/len(_Dd);_rhs=_Dd.T@_td/len(_Dd)
    _Gf=rp_Xa[_tr].T@rp_Xa[_tr]/_tr.sum()
    for _pen in (0.1,1.0,3.0,10.0):
        _B=solve(_Gd+_pen*_Gf+1e-4*np.eye(577),_rhs,assume_a="pos")
        _w=_B[:576]/rp_xsd;_b=_B[576]-rp_xmu@_w
        rp_linear_fits[(_drug,_pen)]=(_w,_b)
        _facts=rp_X[_ho]@_w+_b
        _summary=[]
        for _mult in (0.0,0.5,1.5,2.0,3.0):
            _delta=(rp_cf_input[(_drug,_mult)][_ho]-rp_X[_ho])@_w
            _target=(_mult-1)*ds_rates[_ho,_j]
            _slope=float(_target@_delta/(_target@_target))
            _summary.append(round(_slope,3))
        print(_drug,"pen",_pen,"active factual RMS",round(float(np.sqrt(np.mean(_facts**2))),4),
              "slopes x0/.5/1.5/2/3",_summary)
    _gu=[];_gy=[];_gut=[];_gyt=[]
    for _x in [rp_X]+list(rp_cf_input.values()):
        _Ua=np.column_stack([_x[:,384:],np.ones(len(rp_X))])
        _gu.append(_Ua[rp_train]);_gy.append((2*_active-1)[rp_train])
        _gut.append(_Ua[rp_hold]);_gyt.append((2*_active-1)[rp_hold])
    _gu=np.vstack(_gu);_gy=np.concatenate(_gy)
    _gut=np.vstack(_gut);_gyt=np.concatenate(_gyt)
    _B=solve(_gu.T@_gu/len(_gu)+1e-4*np.eye(193),_gu.T@_gy/len(_gu),assume_a="pos")
    rp_gate_fits[_drug]=(_B[:192],_B[192])
    _score=_gut@_B
    print(_drug,"presence gate held accuracy",float(np.mean((_score>0)==(_gyt>0))),
          "active score p1/50/99",np.round(np.percentile(_score[_gyt>0],[1,50,99]),3),
          "absent score p1/50/99",np.round(np.percentile(_score[_gyt<0],[1,50,99]),3))
with open("/work/notes.md","a") as _f:
    _f.write("\n## Run 175\nFinal-layer contrast decoder failed NE x2 despite reasonable pooled slopes. Tested input-level linear decoders conditional on active drug, with separate presence gates; this distinguishes genuine dose sensitivity from pooled-bin artifacts. No edit installed.\n")
```
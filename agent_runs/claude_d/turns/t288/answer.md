`ctx_of(p, dt)` reads states from the global XALL by row index, so I can rebuild head inputs from the edited states by swapping in an edited copy of XALL. The npz rows are not in (stay, time) order, so I will map them to my STALL rows by (stay, minute), which is unique per bundle. Then:
1. check that the "original" pass reproduces XALL exactly,
2. check that `ctx_of(prev_row, dt)` reproduces my saved head inputs (HI['ctx']),
3. compute with_code under both state sets and summarise |Δwc| by reading type and by minutes since the last NE bundle.

A clear Δwc at map_arterial readings after NE bundles, small elsewhere and at long gaps, means the injection carries to the head.

```python
key=lambda s,m: s*100000+m
P0m=np.round(P0['hour']*60).astype(int); mp={key(s,m):i for i,(s,m) in enumerate(zip(P0['stay_id'],P0m))}
st=STALL[STALL.stay_id.isin(selP)]; idx=np.array([mp[key(s,m)] for s,m in zip(st.stay_id,st.minute)]); rows=st.row.values
print('orig vs XALL max', np.abs(P0['x'][idx]-XALL[rows]).max().round(5), 'edit-orig max', np.abs(P1['x'][idx]-P0['x'][idx]).max().round(3))
ri=np.where(RD.stay_id.isin(selP).values)[0]; dt=(RD.minute.values[ri]-RD.prev_min.values[ri])*60.0
c_o=np.array([ctx_of(int(p),float(d)) for p,d in zip(RD.prev_row.values[ri],dt)]).reshape(len(ri),-1); print('ctx repro err', np.abs(c_o-HI['ctx'][ri]).max().round(5))
XALL_o=XALL; XALL=XALL_o.copy(); XALL[rows]=P1['x'][idx]
c_e=np.array([ctx_of(int(p),float(d)) for p,d in zip(RD.prev_row.values[ri],dt)]).reshape(len(ri),-1); XALL=XALL_o
zo=zpr(c_o); ze=zpr(c_e); zo=zo[0] if isinstance(zo,tuple) else zo; ze=ze[0] if isinstance(ze,tuple) else ze
print('zpr shape', np.shape(zo), 'zmu err', np.abs(np.asarray(zo)[:, :HI['zmu'].shape[1]]-HI['zmu'][ri]).max().round(5))
wo=wc_from(np.asarray(zo)[:, :32],c_o,HI['reg'][ri]); we=wc_from(np.asarray(ze)[:, :32],c_e,HI['reg'][ri]); dW=we-wo
ev=RD.event.values[ri]; nd=np.linalg.norm(dW,axis=1)
for e in np.unique(ev): print(e, 'mean|dwc|', nd[ev==e].mean().round(3), 'sd wc', wo[ev==e].std(0).mean().round(3))
```
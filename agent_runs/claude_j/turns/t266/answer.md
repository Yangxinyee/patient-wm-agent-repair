Run 238 shows the model is already well calibrated. A logit temperature of 1.0 is optimal, so there is no extra factual budget to gain that way.

The action token u is the only route by which dose can reach the state. Next I will measure how sensitive one core step is to u at an insulin bundle. I will add random perturbations of norm 1, 5 and 20 to u and record how much the step's output changes, scaled by u's own norm. If even large changes to u barely move the output, the prelude hardly attends to u, as with z. Then any real fix needs the attention changed. If the output moves substantially, a carefully chosen u direction could carry the dose effect.

```python
torch.manual_seed(0); print('|u| %.2f |e| %.2f'%(float(u_.norm()),float(e_.norm())))
for nrm in [1.0,5.0,20.0]:
    ds=[]
    for t in range(4):
        r=torch.randn(192,dtype=torch.float64); r=r/r.norm()*nrm
        o2,h2=step(o_,h_,dts,e_,u_+r,z0,Mem,ages); ds.append((float((o2-r1[0]).norm()),float((h2-r1[1]).norm())))
    print(nrm,'d_out %.3f d_h %.3f'%tuple(np.mean(ds,0)))
r=torch.randn(192,dtype=torch.float64); r=r/r.norm()*5
o2,h2=step(o_,h_,dts,e_+r,u_,z0,Mem,ages); print('e +5: d_out %.3f d_h %.3f | |out| %.2f |h| %.2f'%(float((o2-r1[0]).norm()),float((h2-r1[1]).norm()),float(r1[0].norm()),float(r1[1].norm())))
```
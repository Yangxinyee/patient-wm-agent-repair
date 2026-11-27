The gated readouts are clearly better. The insulin readout fitted on glucose rows after an insulin bundle reaches R² 0.60 with gain 0.60, against 0.40 ungated. Its unit-gain values stay within −2.1 to +2.7 for 99% of rows, so my linear-range bound of 10 is ample.

The norepinephrine readout on arterial rows stays weak, at R² 0.18. The arterial-versus-cuff gate separates perfectly, with a wide margin of ±0.9 SD.

The crash is a dtype mismatch in my code: the stored original head tensors are evidently float64, and I cast the edited ones to float32. Next I will keep the dtype of `ORIG` and rerun both offline evaluations: units removed (a = 0), and the gated candidate.

I expect gating to bring the cost far below the ungated 0.018. Glucose rows outside the gate should show zero shift and zero cost. Most of what remains should sit on arterial MAP, because of the weak norepinephrine readout.

```python
dt_=ORIG['W0'].dtype; print('ORIG dtype',dt_, ORIG['TR'].dtype)
_ge_old=gated_edit
def gated_edit(a_ins,a_ne,**kw):
    E_=_ge_old(a_ins,a_ne,**kw); return {k:v.to(ORIG[k].dtype) for k,v in E_.items()}
E0g=gated_edit(0.0,0.0); _=eval_edit(E0g,'a=0 (units removed):')
Eg=gated_edit(-1.9,3.0); Dg=eval_edit(Eg,'gated a_ins=-1.9 a_ne=3:')
```
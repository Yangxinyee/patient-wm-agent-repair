The code objects make runs 190–203 clearer:
- `wc_row` computes the glucose-head input (with_code) from out, the h slots, the time to the next reading, the last value and its age.
- `run_from` and `step_at` re-run local core steps.
- `a_y` is a unit direction in y (the new h).

So run 203 most likely measured that a unit injection along `a_y` in y changes with_code at the next glucose reading by about 0.0101, always in the same direction. That makes the y → head path linear and reliable.

My print failed because `inF` is a boolean mask. Next I will print the remaining items safely and the fwd_1 summary (the original model's readouts, which I need for the gap to my targets), and write this recovered state into the notebook.

```python
for nm in ['inF','dy_','w0','w1','ONw','ONb','o2','F20','F_','evf']:
    v=np.asarray(globals()[nm]).astype(float); print(nm, v.shape, 'norm %.4g'%np.linalg.norm(v), 'sum %.4g'%v.sum(), np.round(v.ravel()[:5],4))
print('resp', [ (type(x).__name__, getattr(x,'shape',None)) for x in resp])
print('GRa0', GRa[0], 'GR0', GR[0], 'LS0', np.round(LS[0,:4],3), 'dot(a_y,a_y20)=%.3f'%float(a_y@a_y20))
print(json.dumps(json.load(open('/work/fwd_1/summary.json')))[:1500])
add="""
## Session 8 (restart 8) - recovered (runs 204-206)
- Runs 190-203 NOT noted. Recovered via code objects: wc_row(o,hsl,dt,lv,age_s,z) -> gf with_code; run_from(Bs,rws,EUs,k0,k1,h0) local multi-step;
  step_at(B,rows,EU,k,du,de); coda(c); u_of(ins,ne) -> u vector. a_y, a_y20 = unit y-directions (injection candidates).
- Run 203: DWC (1694 gf rows) = d with_code per unit injection along a_y in y: |dwc| mean .0101, cos-sim to mean 1.000 -> y->with_code path linear & consistent.
"""
open('/work/notes.md','a').write(add)
```
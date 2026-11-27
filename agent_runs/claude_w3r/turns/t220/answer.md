Run 207 shows that `dy_` is almost equal to `a_y`, so the injected direction survives the coda step nearly unchanged. `ONw`/`ONb` are the out_norm gain and bias, and `inF` marks 30 dimensions. `GRa` turned out to be a DataFrame, which caused the error, and the notebook entry was not written. This time I will write the notes first, then print the original model's dose_edit readouts from fwd_1. The gap between those readouts and my record-based targets sets how much effect the edit must add.

```python
add="""
## Session 8 (restart 8) - recovered (runs 204-207)
- Runs 190-203 NOT noted. Recovered via code objects: wc_row(o,hsl,dt,lv,age_s,z) -> gf with_code; run_from(Bs,rws,EUs,k0,k1,h0) local multi-step;
  step_at(B,rows,EU,k,du,de); coda(c); u_of(ins,ne) -> u vector. a_y = unit y-direction (injection cand.), dy_ ~ a_y = resulting y change.
  ONw/ONb = out_norm weight/bias. inF = mask of 30 dims. GRa/GR = DataFrames of gf rows.
- Run 203: DWC (1694 gf rows) = d with_code per unit injection along a_y in y: |dwc| mean .0101, cos-sim to mean 1.000 -> y->with_code path linear & consistent.
"""
open('/work/notes.md','a').write(add)
print(type(GRa).__name__, list(GRa.columns), GRa.iloc[0].to_dict())
s1=json.load(open('/work/fwd_1/summary.json')); print(json.dumps(s1)[:1800])
```
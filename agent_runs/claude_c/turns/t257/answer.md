The stay-level standard errors are large for every edit: N5 is +0.00156 ± 0.0011 and N5+H2.5 is +0.00507 ± 0.00165. Even the u=0 test is ±0.0017. So N5+H2 at +0.0037 ± 0.0014 has roughly a 20% chance of exceeding 0.005 on unseen stays, for only a small readout gain. N5 alone stays safe even at +2 SE.

The head0 boost is therefore not worth the risk; N5 is my fallback. Before settling, I'll check whether N5's x2 response saturates at high insulin doses. If it does, a different numeric factor might do better on the main readout at lower cost. I'll start by locating the dose_edit files and their columns.

```python
fs=sorted(glob.glob('/work/fwd_*/dose_edit_*.csv'), key=lambda f:int(f.split('fwd_')[1].split('/')[0]))
print([ (f.split('/')[2], f.split('_')[-1][:-4]) for f in fs])
print(list(pd.read_csv(fs[0]).columns)[:12])
```
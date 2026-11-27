Run 54 showed that my earlier replica used ReLU as the MLP activation (`act` is `torch.relu`). The other helpers' source is gone, so I can't see how that choice was made.

Before building an edit around ReLU units, I want to know whether ReLU was confirmed or just guessed. `best` and `opts` in the kernel probably hold the configuration search from runs 44–49, with errors for each variant. If ReLU won clearly over GELU or SiLU, I can design repurposed z_post units that are exactly linear when active and exactly zero when inactive. If the activation was never settled, I'll set up a cheap probe through the forward tool to pin it down before relying on it.

```python
def short(v, n=400):
    s = repr(v); return s[:n] + ('...' if len(s) > n else '')
for k in ['best','opts','hinit','oinit','octx','gscale','agg','size','hv','gv','ov','useg','gin','perm','strict','cats','bins0','kq','unit']:
    if k in globals(): print(k, '=', short(globals()[k]))
print('err stats', float(err.abs().mean()) if hasattr(err,'abs') else err)
print('res_ head', short(res_, 800))
```
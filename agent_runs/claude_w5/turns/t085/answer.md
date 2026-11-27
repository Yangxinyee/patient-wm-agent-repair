Run 80 confirms that the alignment is right: the stored z belongs to the same bundle. My predicted z is on the right scale, and some dimensions match well (correlation 0.8–0.9), but others hardly match at all. That points to one wrongly scaled input that affects some directions much more than others. The most suspicious is the treatment of time:
- the decay of h over the gap, where dropping the decay fit better than the documented one;
- the gap input itself.

Both depend on the unit of dt. I'll scan a grid:
- decay strength: dt multiplied by 0, 1/3600, 1/60, 0.1, 0.3, 1;
- gap input: log(dt)/10 in seconds, in minutes, log of hours, or no gap term at all.

A single combination with a sharp drop in error would identify the convention. A flat landscape would point back to the bundle encoding.

```python
def flow_c(h_p, dt, c):
    return h_p * torch.exp(-(dt*c) * torch.exp(T_['core.flow.a'])) if c > 0 else h_p
def Gx(dt, mode):
    if mode == 'none': return torch.zeros(192)
    x_ = {'s10': math.log(dt)/10, 'min10': math.log(dt/60)/10, 'h': math.log(dt/3600), 'h10': math.log(dt/3600)/10}[mode]
    return lin_(Fnn.gelu(lin_(torch.tensor([x_]), 'gap.0')), 'gap.2')
res_tc = []
for c_ in [0, 1/3600, 1/60, 0.1, 0.3, 1.0]:
    for gm in ['s10', 'min10', 'h', 'h10', 'none']:
        r_ = zp_generic(stays_t, lambda o,h,dt,wa, c_=c_, gm=gm: torch.cat([o + ln_(flow_c(h, dt, c_).mean(0),'core.out_norm') + Gx(dt, gm), E_(wa), U0]), max_n=80)
        res_tc.append((r_, c_, gm))
res_tc.sort(key=lambda t: t[0][0]); print(res_tc[:10])
```
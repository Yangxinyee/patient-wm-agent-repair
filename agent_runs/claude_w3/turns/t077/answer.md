The tool's edit names are `insulin_x2`, `insulin_half`, `ne_x2` and `ne_half`. Only the first matched, which is why the other three came back empty.

With the refined impulse response, the true insulin x2 readout on these 815 reached rows is −14.1 mg/dL. My earlier preliminary estimate was −17.5. The model currently gives −0.39. I will rerun with the correct names to get all four targets, plus standard errors if the helper can use the stored covariances. I expect insulin_half to be about +7 (half of −14.1, sign flipped) and the norepinephrine readouts to be small and positive for x2.

```python
for ed_, D_, th_, mu_, V_ in [('insulin_x2',INS_tu,th_ins,2.0,V_ins),('insulin_half',INS_tu,th_ins,0.5,V_ins),('ne_x2',NE_tu,th_ne,2.0,V_ne),('ne_half',NE_tu,th_ne,0.5,V_ne)]:
    try:
        r_ = readout_truth(de1, ed_, D_, th_, mu_, V_)
    except Exception as e_:
        r_ = readout_truth(de1, ed_, D_, th_, mu_); 
    res_t[ed_] = r_
    print(ed_, [np.round(float(x),3) for x in r_])
```
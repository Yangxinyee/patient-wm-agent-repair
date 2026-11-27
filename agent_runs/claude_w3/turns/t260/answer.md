The score is linear: z = (x − median) / IQR per code, clipped to [−10, 10]. A doubled rate therefore moves the z input of `embed.numeric` proportionally, unless the clip binds. The saturation I saw comes from the encoder's nonlinear layers, not from the input. So a dedicated linear path from the rate z-score is feasible.

Next I'll print the per-code medians and IQRs for code 8 and code 9 rates. That tells me how often doubled rates would hit the clip at 10, and what affine map takes z back to a rate.

```python
for e_ in VOCAB['numeric']:
    s_=str(e_)
    if any(t in s_ for t in ["'code': 8","'code': 9","code_id': 8","code_id': 9"]) or VOCAB['numeric'].index(e_)<2:
        print(s_[:300])
print(str(VOCAB['numeric'][0].keys()) if isinstance(VOCAB['numeric'][0],dict) else '')
```
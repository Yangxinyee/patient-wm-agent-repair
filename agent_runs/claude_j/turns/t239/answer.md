Run 211 shows that my notebook has no record of the amplification test, so I don't know how it came out.

Here is where I stand. The edit can only read the head inputs, and those carry little dose information. A ridge readout on them either misses most of the true effect or costs too much factual fit. To fix this, I probably need to edit the recurrent core so that the state carries a clean, slowly decaying dose signal the head can read.

Next I will read the section headers of ARCHITECTURE.md and the lines about the core update. I am looking for a linear injection point, such as the `b_in` path or a residual stream that skips the normalisations. Finding one would confirm the plan. If every path runs through normalisations, I will fall back to a head-only edit, fitted on counterfactual data from more stays.

```python
A_=open('/model/ARCHITECTURE.md').read()
ls=A_.split('\n'); print(len(A_),len(ls))
print('\n'.join(l[:120] for l in ls if l.startswith('#'))[:1500])
```
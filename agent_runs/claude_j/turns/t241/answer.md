Run 213 shows that every path into h passes through normalised transformer layers: the prelude, then the blocks, then `c_out` and the coda. A clean linear dose channel in the core is therefore hard to build. The input end is still a usable lever. The dose numerics enter the action embedding linearly, and the kernel's `set_amp` / `amp_eval` seem to scale exactly that.

Next I will read the constants and variable names of `set_amp` and `amp_eval`. I want to know which weights they scale, how the model is restored afterwards, and what `amp_eval` reports, before calling it. If `set_amp(1.0)` restores the original weights and the output is a short summary of the counterfactual shift, I can test amplification cheaply on a few stays.

```python
for n in ['set_amp','amp_eval','fact_nll','gen_stay','struct_delta','drug_hist']:
    c=globals()[n].__code__
    print(n,'| C:',[x for x in c.co_consts if isinstance(x,(str,int,float)) and x is not None][:14],'| V:',c.co_varnames[:16])
print(type(NW0),getattr(NW0,'shape',None),type(NW0W),getattr(NW0W,'shape',None))
```
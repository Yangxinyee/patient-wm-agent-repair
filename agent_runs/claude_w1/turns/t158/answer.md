The pipeline is clear. gen_samples(df) builds bundles and runs the exact core with the global weights Wo, then builds the head inputs with wc_vec. head_mean(rows) turns rows into predicted means. So I can test a candidate weight edit locally by patching Wo and comparing predicted means against the unpatched run.

Prelude weights: kv.weight (384×192) holds K in rows 0–191 and V in rows 192–383, plus a 192×192 out projection. Bundles carry the encoded e and u vectors. Before writing the injection I need two facts:
- how run_stay's edit argument is used (it may already be a hook);
- what run_stay returns.

I also want to check the prelude attention pattern on reading bundles, not only drug bundles. The injection only behaves cleanly if slots attend almost entirely to the u token on drug bundles and the empty-set u token gets a zero decoder output. This run disassembles around 'edit' and prints the state structure.

```python
import dis, io
buf = io.StringIO(); dis.dis(run_stay, file=buf); L = buf.getvalue().splitlines()
hits = [i for i,l in enumerate(L) if "(edit)" in l]
print('edit refs', len(hits))
for i in hits[:3]:
    print(' | '.join(x.strip()[-38:] for x in L[max(0,i-2):i+6]))
St0 = run_stay(B0)
print(type(St0), len(St0) if hasattr(St0,'__len__') else '')
e0 = St0[0] if isinstance(St0,(list,tuple)) else St0
print(type(e0), (list(e0.keys()) if isinstance(e0,dict) else np.shape(e0)))
```
The row norms show the embedding composition is right:
- Unused rows sit at the untrained initialisation norm of about 0.28, and row 0 of the small categorical tables is zero.
- So the missing tables add nothing. `kq` is always row 0.
- The units are consistent: mg/dL is row 2 and mmHg row 5, both heavily used.

That leaves value-bin ties (readings and bin edges are both integers) or a wrong unit assignment as the likely e bug.

I'm not sure an exact e encoder is even needed, though. The key place for a repair is the drug bundle at hour + 5 minutes:
- It is the last bundle before each reading.
- It carries the lag-1 dose in u_k and the lag-2 dose in its END atom, which my e already nearly matches.

If my core step reproduces the tool's states at drug bundles, given the previous state, my e and u, and the tool's z, I can compute exactly how u and z move the next glucose prediction. To check that, I'll look at the states file layout and my `core_step` code.

```python
npz=np.load('/work/fwd_2/states_original.npz', allow_pickle=True)
print({k:(npz[k].shape,npz[k].dtype) for k in npz.files})
print(json.load(open('/work/fwd_2/summary.json')))
src2=open('/work/wm.py').read(); i=src2.find('def cross_mem'); print(src2[i:i+2200])
```
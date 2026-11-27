The helper confirms that the edit would act separately on all four slots. Its verbose output prevented the slot-level variance check from appearing. I’ll rerun that check concisely and inspect the required replay inputs. Small residuals in every slot would support a pilot; large residuals hidden by slot averaging would reject it.

```python
print("PER-SLOT FACTUAL SCORES")
score139 = (cache136[1].double() - torch.tensor(mu136)) @ torch.tensor(B136)
print("means:\n", score139.mean(0).numpy())
print("SDs:\n", score139.std(0).numpy())
print("maximum absolute:\n", score139.abs().amax(0).numpy())
print("SD after averaging slots:", score139.mean(1).std(0).numpy())
for gain140 in [0.001, 0.01, 0.1]:
    inj140 = score139 @ torch.tensor(directions136.T) * gain140
    E140 = layer_norm(cache136[2] + inj140.float(), 'core.prelude_norm')
    print("gain", gain140, "injection RMS", float(inj140.square().mean().sqrt()),
          "E change RMS", float((E140-cache136[0]).square().mean().sqrt()))
print("\nREPLAY INPUT INVENTORY")
for k in sorted(list(globals())):
    if k.endswith('136'):
        v = globals()[k]
        if isinstance(v, (np.ndarray, torch.Tensor)):
            print(k, tuple(v.shape))
        elif isinstance(v, (int,float,np.number)): print(k, v)
        elif isinstance(v,str) and len(v)<100: print(k,v)
print("\nSHORT HELPER REFERENCES")
for k in ['state_update82','next_view105','readout134','memory82']:
    f = globals()[k]
    print(k, "names", f.__code__.co_names, "constants",
          [x for x in f.__code__.co_consts if isinstance(x,(str,int,float))])
print("CURRENT EDITS", edited())
```

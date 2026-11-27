Run 230 shows the patch does fire. `zpost` was called 79 times and matched an insulin bundle 25 times. Yet run 229 showed no change in the factual predictions.

My guess is that the replica's factual run takes z from the stored tool states rather than from `zpost`, and only the counterfactual run recomputes it. Next I will compare the factual and the counterfactual logits between no injection and a −30 mg/dL injection on one stay. If only the counterfactual logits change, the guess is right, and I will test the lever on the counterfactual run instead.

```python
outs={}
for iz in [0.0,-30.0]:
    INJ_Z=iz; CACHE.clear(); INS_KEY.clear(); outs[iz]=amp_eval(1.0,[s0])[0]
a0,a1=outs[0.0],outs[-30.0]
print('lgf diff %.4f lgc diff %.4f'%(float((a0[3]-a1[3]).abs().max()),float((a0[4]-a1[4]).abs().max())), 'k0,k1',a0[1],a0[2])
INJ_Z=0.0
```
`run_stay_z` returns, for each bundle of the stay, the four h slots, the out vector and z. I'll compare each against the matching part of the stored tool states (out = first 192 numbers, h = next 768, z = last 32).

I'll also print `head2.py`, which is short and readable. `value_mean` there is how I turn a state into a predicted glucose or MAP mean, and I need its exact inputs before measuring how a z injection moves the next reading's prediction.

```python
H_, O_, Z_ = [N(x) for x in out]
print('err h', np.abs(H_.reshape(len(ks0),-1)-X3n[ks0][:,192:960]).max().round(5), 'out', np.abs(O_-X3n[ks0][:,:192]).max().round(5), 'z', np.abs(Z_-X3n[ks0][:,960:]).max().round(5))
print(open('/work/head2.py').read())
```
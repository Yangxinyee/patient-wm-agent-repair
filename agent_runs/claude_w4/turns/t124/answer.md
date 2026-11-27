Both state files cover the same 5,218 bundles in the same layout: 192 numbers for out, then 4×192 for h, then the 32 z means. I can compare them directly.

Now I compare the original states (fwd_3) with the probe states (fwd_4, where z_in.weight was 0). The question is how much removing the z token changed out and h, relative to how much those states vary across bundles. A small ratio (around 0.01) means the z route has almost no leverage and the injection must go elsewhere. A ratio of 0.3 or more means z matters and the v1 failure lies in the gating or scale.

```python
xa, xb = A['x'], B['x']
def rel(sl):
    d = xb[:,sl]-xa[:,sl]; s = xa[:,sl].std(0).mean()
    return np.abs(d).mean()/s, np.sqrt((d**2).mean())/np.sqrt(xa[:,sl].var(0).mean())
print('out', rel(slice(0,192)))
for j in range(4): print('h',j, rel(slice(192+192*j, 384+192*j)))
print('z', rel(slice(960,992)))
st = A['stay_id']; hr = A['hour']
print('first stay rows', (st==st[0]).sum(), 'hours range', hr.min(), hr.max())
```
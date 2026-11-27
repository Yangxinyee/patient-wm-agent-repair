The fresh replica run agrees with `EDR`. Doubling insulin from hour 12 changes the value-head input by only 0.09 in norm, and the change points in inconsistent directions across rows (mean cosine 0.41). The ridge decoder responds by just 0.05 per +1.2 dose units. So the original state hardly encodes the dose by the time it reaches the head, and no readout-only edit can produce a large, clean effect.

The dose has to be written into the state upstream. It is cleanest in the prelude's u token, which head 2 attends to fully and where the dose is linearly decodable with R² 0.999. A value-head unit can then read it back and convert it into a glucose (or MAP) shift. To plan where to inject, I'll list the core and prelude tensors and their shapes.

```python
sel = [nm for nm in NAMES if nm.startswith('core.') and ('prelude' in nm or 'b_in' in nm or 'delta' in nm or 'a_in' in nm or 'c_out' in nm or 'norm' in nm)]
print(len(sel)); print([(nm.replace('core.',''), SHAPES[nm]) for nm in sel][:40])
```
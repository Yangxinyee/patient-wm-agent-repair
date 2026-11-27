The vocabulary shows that every reading and dose is scored per code as z = (x − median)/IQR, clipped to [−10, 10], plus a fine bin from 1 to 16. For insulin the median rate is 2 units/h with IQR 2. For norepinephrine the median rate is 0.53 mcg/kg/min with IQR 0.62, and its upper bins all collapse at a rate of 1.0, so the recorded rates seem to be capped at 1.0. A ×2 edit therefore changes z_rate and z_dose and moves the bin. Above the cap, only z changes because the bin is already the last one.

Next I will print the rest of the numeric table and the tensor list with shapes. I need to know which tensors handle drug atoms (embed.numeric, embed.bins.rate and .dose, encode_action.*) and whether they are shared with the readings.

```python
for n in VOC['numeric']:
    print(n)
print(VOC['numeric_note'])
print(len(NAMES))
for n in NAMES:
    print(n, SHAPES[n])
```
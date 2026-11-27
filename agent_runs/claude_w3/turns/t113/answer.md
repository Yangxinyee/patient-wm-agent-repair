The raw file has one row per event, and an infusion record carries `end_hour`. The model adds END atoms itself (kind 2) at those end times, so my bundle builder must create them correctly. Next I'll check whether my bundle table for that stay contains kind-2 atoms, and where they fall relative to infusion starts.

```python
print(b0['kind'].value_counts().to_dict())
print(b0[(b0.minute>=0)&(b0.minute<=130)].to_string()[:1500])
print('n atoms per bundle (dump) first 10:', F4['n_atoms'][:10], 'hours:', F4['hour'][:10])
```
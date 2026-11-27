## Step 2: units, timing and the full vocabulary

**What the last result showed.**
- The data are hourly. Readings fall on integer hours; insulin records start at hour + 0.0833 (5 minutes after a reading) and last one hour.
- Each drug has exactly one unit: insulin in units/hour, norepinephrine in mcg/kg/min. So unit quirks are not a source of exogenous variation.
- There are 30,000 stays and about 2M events.
- The value-bin centres show the predicted mean is a sum over 16 discrete bin centres. Its resolution is therefore limited by the bin spacing, which matters for small effects.

**What I need next.**
- The full numeric table: the rate and dose medians, IQRs and bin edges for insulin and norepinephrine. A dose edit changes z_rate, z_dose and the bins, so I need to know exactly which inputs a doubling moves.
- The code list and the tensor names.
- The distribution of rates: are they discrete steps, and how do they relate to the preceding reading?
- Stay lengths and how often each drug is given.

**What would count as evidence.** A small set of discrete rates (for example 1, 2, 3 units/h) that switches at fixed reading thresholds would point to a sliding-scale protocol and a discontinuity design.

```python
print(json.dumps(voc['codes'], indent=0)[:3000])
for e in voc['numeric']:
    print(e)
print(voc['numeric_note'])
tn = json.load(open('/model/tensors.json'))
print(type(tn), len(tn))
for n in NAMES:
    print(n, SHAPES[n])
```
In the tuning set, 222 of 400 stays get insulin from hour 12 on and 153 get norepinephrine. Only 86 of those norepinephrine stays have arterial readings after hour 13, and the norepinephrine readout is measured on map_arterial. So the main readout's sample is mainly insulin stays, and the norepinephrine readouts rest on stays with an arterial line.

Next I'll spend one forward pass on a dose_edit with the original model over 100 informative tuning stays: those with insulin from hour 12, plus norepinephrine stays with arterial readings, prioritising stays with both. This gives the output format, the exact averaging, and the model's current answers. I'll then compare those with my g-estimation-based truth computed on the same stays.

```forward
{"type": "dose_edit", "model": "original", "split": "tuning", "stays": [2020000000]}
```

Correction: I want an informative set of 100 stays rather than one. I'll first select them in Python, so this block is replaced by the selection below.

```python
both_ids = sorted(set(tu_ins12.index) & set(tu_ne12.index) & set(tu_art.index))
ins_only = sorted(set(tu_ins12.index) - set(both_ids))
ne_only = sorted((set(tu_ne12.index) & set(tu_art.index)) - set(both_ids))
sel = both_ids[:]
k1 = (100 - len(sel))//2
sel += ne_only[:k1]
sel += ins_only[:100-len(sel)]
sel = [int(x) for x in sel]
print(len(both_ids), len(ne_only), len(ins_only), len(sel))
json.dump(sel, open('/work/sel_tuning_A.json','w'))
print(json.dumps(sel))
```
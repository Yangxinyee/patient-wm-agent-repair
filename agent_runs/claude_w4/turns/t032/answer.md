## Step 30 (redo): get the stay IDs for the first forward pass

**What happened.** My last reply had two code blocks, so nothing ran. I also used placeholder stay IDs in the forward request, which would have been wrong.

**What I will do next.** Print and save the IDs of the 100 tuning stays that have both insulin and norepinephrine from hour 12. Then I will request the baseline dose_edit on exactly those stays.

**What would count as evidence.** This step only prepares the sample, so there is no hypothesis to test yet.

```python
both_ids = info[(info.ins_12on>0)&(info.ne_12on>0)].index.tolist()
sel100 = [int(s) for s in both_ids[:100]]
print(len(both_ids))
print(json.dumps(sel100))
json.dump(sel100, open('/work/sel100_both.json','w'))
```
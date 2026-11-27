The v3 edit (gI = 1, gN = 1, sP = 0) is saved. It touches only head-1 value rows 240–287 and head-1 output columns 48–95 of the first prelude attention, and a copy is kept in /work.

To send the dose_edit request I need the exact list of the 100 tuning stays (sel100) that the truth figures were computed on. Only its beginning is visible in this conversation, so I'll print the full list compactly, along with a quick check that it matches `sel100` in the kernel.

```python
import json
ids=json.load(open('/work/sel100_both.json'))
print(ids==list(map(int,sel100)) if 'sel100' in globals() else 'no sel100')
print(json.dumps([int(i) for i in ids]))
```
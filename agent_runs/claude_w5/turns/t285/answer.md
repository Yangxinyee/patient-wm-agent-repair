The kv edit is saved: the tensor matches `WE` to float32 rounding (6e-5 on entries up to 1928), and the edit now covers 5 tensors. The forward request needs the full list of stay IDs, which I only printed in part.

I'll print the full tuning selection A, the 100 stays used in fwd_1 and fwd_4. I'll also print the training selection B (the 100 stays of fwd_2 and fwd_3), which I will need later for the factual-NLL check.

```python
import json
selB_list = [int(s) for s in (lambda x: x if isinstance(x, list) else x.get('stays', x))(json.load(open('/work/sel_train_B.json')))]
print('A', json.dumps(selA_list))
print('B', json.dumps(selB_list))
```
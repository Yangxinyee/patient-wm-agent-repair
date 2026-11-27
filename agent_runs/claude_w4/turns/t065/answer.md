The probe edit is saved: z_post now copies 30 chosen input coordinates (4 ctx, 6 e_k, 20 u_k), plus two activation-test outputs, into the posterior mean, and z_in is zeroed.

To send the states request I need the literal stay ids of the 100 tuning stays (sel100) used in earlier forward passes. Reusing them lets me compare with the original model's states in fwd_3. I'll print them compactly. No hypothesis is tested here.

```python
import json
print(len(sel100)); print(json.dumps([int(s) for s in sel100]))
```
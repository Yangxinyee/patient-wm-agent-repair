My saved settings disagree with the guesses I just tested. They say the MLPs use GELU, not SiLU. The unit index is mmHg → 2 and mg/dL → 3. END atoms keep their rate and dose. With SiLU the error was 0.24. I'll repeat the check with exact GELU and tanh-GELU, using the saved unit map.

```python
import math
ACTF = {'gelu': F_.gelu, 'gelu_t': lambda x: F_.gelu(x, approximate='tanh')}
def mlp_(x, p, Pp, act=None): 
    a = ACTF[act or CUR_ACT]
    return lin_(a(lin_(x, p+'.0', Pp)), p+'.2', Pp)
UNIT = {'mmHg':2, 'mg/dL':3, 'units':4, 'mg':5}
for CUR_ACT in ['gelu', 'gelu_t']:
    print(CUR_ACT, np.round(check(dict()), 4))
```
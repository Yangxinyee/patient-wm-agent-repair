Run 80 shows the value head is a plain two-layer MLP on `with_code`: Linear 192→192, GELU, Linear 192→17, with no inner LayerNorm. The `embed_transition.value_bin` table has 13 codes × 17 rows.

Factual noise scale on 100 training stays:

| Event | Share of readings | MAE | Implied sd |
|---|---|---|---|
| glucose_finger | 33% | 11 mg/dL | ≈14 mg/dL |
| map_arterial | 44% | 4 mmHg | ≈5 mmHg |

The overall NLL is 1.44.

What this means for the edit:
- **NLL budget:** shifting predicted means by noise of sd ≈1.5 mg/dL (glucose) or ≈0.5 mmHg (MAP) each costs about 0.002 nats per reading overall. The 0.005 budget allows both, but only just.
- **Cleanest edit site:** a rank-1 or rank-2 change to `attr_heads.value_bin.2`, of the form W2 += tilt ⊗ v. Here v is a probe fitted by ridge on the head's own hidden GELU activations. Its targets would be code-specific: the dose effect on glucose_finger rows, the norepinephrine effect on map_arterial rows, and 0 on other codes. This leaves the core untouched.

Before building it I need my existing local head code: its functions and saved arrays. Then I can compute hidden activations and test whether they decode the insulin rate or deviation better than the raw state did (R2 0.67 / 0.24).

```python
import inspect, re
src = open('/work/impl2.py').read()
print(len(src))
for m in re.finditer(r'^def (\w+)\(.*$', src, re.M): print(m.group(0))
for nm in ['head_inputs','Head']:
    obj = globals().get(nm)
    try: print('-----', nm); print(inspect.getsource(obj)[:2500])
    except Exception as e: print(nm, type(obj), e)
for nm in ['FH','DH','PMm','SX','SST','SHR','fac','recs','HD','hid','wc','zm','st_','ip','ip2','rows_h','fsa','key_k']:
    obj = globals().get(nm)
    print(nm, type(obj).__name__, getattr(obj,'shape', (len(obj) if hasattr(obj,'__len__') else '')))
```
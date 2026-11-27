The last listing showed that the value head `attr_heads.value_bin` is a small MLP: a 192→192 layer, a nonlinearity, then 192→17. `last_numeric` takes 8 inputs. Both matter for where an injected signal can enter. Because the head is an MLP, adding a vector along the z_last_value column of `last_numeric` would make the head respond as if the last glucose value were shifted. That is a natural lever for a dose-dependent shift.

Next I will save this to the notebook and print the stored impulse-response estimates (`th_ins`, `th_ne`), rounded and short. These are the targets the edit must reproduce. The check: they should match the lag profiles written in my notes.

```python
import inspect
with open('/work/notes.md','a') as fh:
    fh.write("\n## Session 3\n- value_bin head = MLP 192->192->17; last_numeric (192,8). Lever: add vector along last_numeric z_last_value column scaled by a dose memory.\n")
print('th_ins', np.round(np.asarray(th_ins),2))
print('th_ne', np.round(np.asarray(th_ne),2))
print('readout_truth', inspect.signature(readout_truth))
print('L_numeric cols norm', np.round(load('last_numeric.weight').norm(dim=0).numpy(),2))
```
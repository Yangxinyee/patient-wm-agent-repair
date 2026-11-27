Run 81 confirmed two things. The two current operations scale the z_dose and z_rate columns by 20 (alpha 19), which pass 9 showed is worse than x5. And the dose must reach the state mainly through the action token u_k in the prelude layer: u_k is zero at reading bundles, and z posterior amplification did nothing.

Next I will list the tensor names and shapes for the prelude, the action encoder and the value head. This should show whether there are separate tensors where an edit would touch only the drug-bundle pathway. If there are separate q/k/v projections, or encode_action layers after the inputs, that gives a lever downstream of the saturating numeric inputs. If not, I need a different route.

```python
grp = {}
for n in NAMES:
    p = '.'.join(n.split('.')[:2])
    grp[p] = grp.get(p, 0) + 1
print(grp)
for n in NAMES:
    if n.startswith(('core.prelude', 'encode_action', 'core.b_in', 'core.z_in', 'attr_heads.value')) or 'value_bin' in n:
        print(n, SHAPES[n])
```
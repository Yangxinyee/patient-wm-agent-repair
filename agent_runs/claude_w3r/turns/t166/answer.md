The z-input projection isn't stored as `core.z_in.*`. I'll list the tensor names containing "z_in" or similar (likely a small MLP with `.0`/`.2` layers) and adapt the token construction, then rerun the attention check. Success is still a match between my attention weights and `step_parts`' attention weights.

```python
print([n for n in NAMES if 'z_in' in n or 'zin' in n or n.startswith('core.z')])
```
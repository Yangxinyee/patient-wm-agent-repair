The saved states cover dose bundles (pass 3) and every bundle for 12 stays (pass 4), so I can take the previous state for each bundle from pass 4. Before re-validating the local step I'll list the tensor names and shapes for the dose encoder, the posterior, `z_in` and the prelude, so I know exactly what the computation uses.

```python
sel = [n for n in NAMES if re.match(r'(encode_action|core\.z_post|core\.z_in|core\.prelude|core\.slots|embed\.numeric|core\.gap)', n)]
print(len(sel))
print('; '.join(f"{n}:{tuple(SHAPES[n])}" for n in sel))
```
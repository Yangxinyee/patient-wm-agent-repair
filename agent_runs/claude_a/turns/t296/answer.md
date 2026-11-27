Run 282 showed that the insulin repair works in my offline copy of the model. With theta = −2.45:

| Readout | Edited model | My estimate of the truth |
|---|---|---|
| insulin ×2 | −14.93 | −14.98 |
| insulin ×½ | +7.19 | +7.49 |
| norepinephrine ×2 | −0.04 | +3.64 |
| norepinephrine ×½ | −0.32 | −1.82 |

- **Factual cost:** +0.0006 nats on the drug-heavy train set and +0.0001 on the general set. Both are well inside the 0.005 limit.
- **Saved:** this version is saved as the current edit (71 tensors).
- **Norepinephrine:** not repaired yet. Its two readouts are still far from the truth.

I only have 18 Python runs left but 46 forward passes, so the order matters:
1. **Check the saved edit with the real tool.** My offline copy was only checked against the tool for the original model. Running dose_edit and factual on the 100 tuning stays of sel1 would test the edited model directly. My estimates of the true effects for those stays are −14.30, +7.15, +4.08 and −2.04.
2. **Then try norepinephrine with the same method.** This depends on my library code being general enough to reuse.

This run collects what I need for both steps:
- the sel1 stay ids, which I have to type into the forward request;
- the function signatures in /work/lib1–4.py and the functions in the kernel;
- a restart entry in the notebook.

What would confirm or refute my current belief:
- **Confirm:** the tool gives about −14 and +7 for the edited model on sel1, and the factual NLL difference stays under about +0.001.
- **Refute:** a large gap from those numbers would mean the tool computes something differently from my offline copy, for example in numerical precision, and my edit depends on that difference.

```python
import json, re, inspect
with open('/work/notes.md','a') as f:
    f.write('\n## Restart 8 (run 283-)\n- Saved edit = insulin fix theta=-2.45 (71 tensors; offline 40 sel1 stays: ins_x2 -14.93 vs truth -14.98, half 7.19 vs 7.49; NE untouched -0.04 vs 3.64, -0.32 vs -1.82). Plan: verify with tool (dose_edit + factual, both, sel1), then attempt NE with same machinery if runs allow; fallback = current insulin-only edit.\n')
sel = json.load(open('/work/sel.json'))
print({k: (len(v) if hasattr(v,'__len__') else v) for k, v in sel.items()})
print('sel1', json.dumps(sel['sel1']))
for fn in ['lib1.py','lib2.py','lib3.py','lib4.py']:
    src = open('/work/'+fn).read()
    print(fn, len(src), re.findall(r'^def (\w+)\((.*?)\):', src, flags=re.M))
g = [k for k, v in list(globals().items()) if inspect.isfunction(v) and getattr(v,'__module__',None) == '__main__']
print(len(g), g)
```
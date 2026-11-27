The architecture confirms that a prelude FFN edit would enter the recurrent state and affect all predictions, not just the glucose/MAP heads. The diagnostic output was truncated, so I still need the exact saved edit and causal estimates. I’ll retrieve those compactly; a low-variance factual residual with consistent dose contrasts would justify a small pilot.

```python
print("EARLY NOTES\n", open('/work/notes.md').read()[:6500])
print("\nPRELUDE TENSORS")
for k in NAMES:
    if 'prelude' in k:
        print(k, SHAPES[k])
print("\nREPLAY AND READOUT HELPERS")
for k in sorted(list(globals())):
    v = globals()[k]
    if inspect.isfunction(v) and getattr(v, '__module__', None) == '__main__':
        try: sig = str(inspect.signature(v))
        except Exception: sig = "?"
        print(k, sig)
print("\nPILOT SUMMARY")
for k in ['B136','directions136','mu136','T136','K136','G136','scale136',
          'scales136','sensors136','b135','fits135','muF134','cols136','keep136']:
    if k not in globals(): continue
    v = globals()[k]
    if isinstance(v, (np.ndarray, torch.Tensor)):
        a = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
        print(k, a.shape, "norm", np.linalg.norm(a), "first", a.reshape(-1)[:10])
    else:
        print(k, str(v)[:600])
print("\nSAVED CODE STRINGS")
for k in sorted(list(globals())):
    if isinstance(globals()[k], str) and any(t in k.lower() for t in ['code13','src13','script13']):
        print(k, globals()[k][-3500:])
print("\nCURRENT EDITS", edited())
```

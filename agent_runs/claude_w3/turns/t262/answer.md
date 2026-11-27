The kernel holds many helper functions; the edit inspection failed only because `load` returns torch tensors. I'll convert to numpy and repeat the inspection so I can see which rows and columns were changed.

```python
def npy(t):
    return t.detach().cpu().numpy() if hasattr(t, 'detach') else np.asarray(t)
for n in edited():
    a = npy(load(n)); b = npy(load_original(n))
    d = np.abs(a - b)
    if a.ndim == 2:
        rows = np.where(d.max(1) > 0)[0]; cols = np.where(d.max(0) > 0)[0]
        print(n, a.shape, "rows", rows[:6].tolist(), len(rows), "cols", cols[:6].tolist(), len(cols), "max", round(float(d.max()), 3))
    else:
        idx = np.where(d > 0)[0]
        print(n, a.shape, "idx", idx[:6].tolist(), len(idx), "new", a[idx[:3]].round(3).tolist(), "old", b[idx[:3]].round(3).tolist())
```
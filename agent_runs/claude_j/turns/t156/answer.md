`T` is only the transpose attribute, so the replica has no hidden flags. The mismatch from bundle 5 on must come from how memory is filled (which bundles enter M, and with what vector) or from the ages. The architecture file defines what goes into M, so I'll read its short definitions section (lines 18–34) rather than guess further.

```python
for i in range(18,35): print(i, arch_[i][:200])
```
No insulin bundles matched, so the keys of the hourly maps aren't in the form I assumed. I'll print a couple of keys from each map and a few bundle hours, and then match them correctly.

```python
print('I keys', list(I_.items())[:3]); print('G keys', list(G_.items())[:3])
print('hours of insulin bundles', hrs[hi][:6], 'sid', sid[hi][:3], type(sid[0]))
```
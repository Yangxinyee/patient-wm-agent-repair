The bin edges are not stored under that key, but the kernel has an fbin() helper from earlier sessions. I will check its signature and test it on a few observed values. Then I will compute replica factual NLL per code on the Xf rows.

```python
print(inspect.signature(fbin), fbin.__code__.co_names[:12])
print(str(VOCAB['value_bin_note'])[:300])
```
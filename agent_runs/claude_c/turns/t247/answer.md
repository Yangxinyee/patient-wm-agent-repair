Since this is a restart, I'll first recover what my earlier analysis found. The truncated notebook is missing the estimated true effects and the architecture summary, and I need both before deciding about the 50 operations still in place. Those 50 are the numeric-x5 plus head0-x2.5 combo, which was over the factual budget at +0.0051. I'll print the start of the notebook in a short chunk, because long outputs were blocked before.

```python
s=open('/work/notes.md').read()
print(len(s))
print(s[:4500])
```
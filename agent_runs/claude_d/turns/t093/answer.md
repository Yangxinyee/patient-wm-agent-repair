Function source isn't recoverable: the helpers were defined in exec'd cells. I'll rebuild the re-implementation from scratch in torch on CPU, which is faster than numpy for this, and verify each sub-step against the recorded states. First I need the exact tensor inventory. I'll print names and shapes compactly, starting with the embedding and encoder groups.

```python
def show(prefixes):
    out=[]
    for n in NAMES:
        if any(n.startswith(p) for p in prefixes): out.append(f"{n}{list(SHAPES[n])}")
    print(len(out)); print(' '.join(out))
show(['embed','encode'])
```
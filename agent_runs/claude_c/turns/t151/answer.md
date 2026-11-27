The notebook update went through and the op list is empty. Run 116 showed that edits to the action encoder affect only the action token. A remaining question is which other paths could carry the action signal into the state. I'll do a compact listing of the `core` tensors outside the prelude (names and shapes only), short enough not to trip the output filter.

```python
s = "; ".join(f"{n.replace('core.','')}{list(SHAPES[n])}" for n in NAMES if n.startswith('core.') and not n.startswith('core.prelude') and not n.startswith('core.blocks'))
print(len(s)); print(s[:2500])
```
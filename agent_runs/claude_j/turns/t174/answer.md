The heads section confirms that the value head reads `with_code`, made from the state plus a same-code "last value" register, and nothing else. The dose-edit section confirms the readout is the head's predicted mean of glucose_finger (bin-centre average) in hours 13–24. It is averaged over 20 free-running paths, and only insulin atoms are changed.

So every dose effect has to pass through the state. The edit must make the state-to-head pathway respond to dose departures at the true strength, while staying near zero on recorded data.

To get counterfactual states I need a replica that is trustworthy on drug bundles, and the old kernel functions are undocumented black boxes. I will rebuild the replica cleanly in a file. As a first step I will print the vocab keys and the `embed.*`/`encode.*` tensor shapes, kept short.

```python
print(list(VOCAB.keys()))
print({k:(type(v).__name__, (list(v.keys())[:8] if isinstance(v,dict) else len(v) if hasattr(v,'__len__') else v)) for k,v in VOCAB.items()})
print(' '.join(f"{n}{list(SHAPES[n])}" for n in NAMES if n.startswith('embed') or n.startswith('encode.') or n.startswith('core.z') or 'z_' in n)[:1500])
```
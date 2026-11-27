The full recurrent reconstruction is not accurate enough to trust absolute local predictions. An earlier experiment did recover an exact GPU-anchored coda input, which is suitable for measuring small output changes. I’ll use that anchor to calibrate a MAP-specific direction, then test the resulting edit on the GPU rather than relying on local reconstruction alone.

```python
print("CODA ANCHOR CONTRACTS")
for k251, v251 in list(globals().items()):
    if k251.endswith("134") and not k251.startswith("_"):
        if hasattr(v251, "shape"):
            print(k251, v251.shape)
        elif isinstance(v251, (float, int, str, tuple)):
            print(k251, repr(v251)[:200])
print("\ncoda_input134")
compact248(coda_input134, 1600)

print("\nSMALL-CONTEXT READOUT")
compact248(next_means45, 3000)
print("\nRELEVANT SMALL-CONTEXT GLOBALS")
for k251 in sorted(set(next_means45.__code__.co_names)):
    v251 = globals().get(k251)
    if hasattr(v251, "shape"):
        print(k251, v251.shape)
    elif isinstance(v251, (dict, list, tuple)):
        print(k251, type(v251).__name__, len(v251))
print("reading masks", int(map_mask82.sum()), int(glucose_mask82.sum()))
print("gain range", as_numpy194(gain81).min(), as_numpy194(gain81).max())
print("previous MAP direction response",
      "MAP mean", np.mean(response82[map_mask82]),
      "glucose RMS", np.sqrt(np.mean(response82[glucose_mask82]**2)))
print("\nMEMORY ENCODING REFERENCES", memory133.__code__.co_names)
print("\nPRELUDE RETURN", prelude_hidden93.__code__.co_consts[-6:])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 251: anchored forward-difference route\n"
            "Full recurrent absolute reconstruction is not trusted (prior best raw h RMSE "
            "5.77); the coda inversion has exact GPU replay. Recovering this anchor and "
            "small-context reading interface for a MAP-specific prelude response direction. "
            "Joint236 unchanged. Python runs: 251; forwards: 29.\n")
```

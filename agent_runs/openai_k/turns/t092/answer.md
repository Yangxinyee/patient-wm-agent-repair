The saved latent directions are weak, but that does not establish that every direction entering the state is weak. I’ll test the input to `z_in` separately from the prior readout and verify the dose decoder’s units. This will support a more direct edit only if decoded rates remain accurate across multipliers and the state can transmit their effects.

```python
print("HELPER SIGNATURES")
for nm86 in ["lin0", "mlp0", "r59_features", "r59_core", "arr71", "action26"]:
    if nm86 in globals():
        print(nm86, inspect.signature(globals()[nm86]))
print("\nSMALL DATASET CONTRACT")
small86 = dataset80(idx80[:3])
def summary86(v):
    if hasattr(v, "shape"):
        return (type(v).__name__, tuple(v.shape))
    if isinstance(v, (tuple,list)):
        return [summary86(x) for x in v]
    if isinstance(v,dict):
        return {k:summary86(x) for k,x in list(v.items())[:5]}
    return type(v).__name__
print(summary86(small86))
print("idx80:", idx80[:8])
print("rate66 samples:", rate66[idx80[:8]])
print("rtrue66 samples:", rtrue66[idx80[:8]])
print("index66:", index66[:12], index66[4035:4047])
print("dr66:", dr66[:8], dr66[4035:4043])
print("dose decoder biases:", decoder_bias66)
print("reserved first-layer rows, nonzero block norms:")
for j86 in reserve66:
    print(int(j86), [float(np.linalg.norm(np.asarray(W166[j86, a:a+192])))
                    for a in (0,192,384)], float(b166[j86]))
print("\nCARRIER RESULT STRUCTURE")
for strength86,obj86 in carrier_results82.items():
    print(strength86, summary86(obj86))
print("\nDYNAMIC TARGETS")
for sensor86,obj86 in dynamic_ir.items():
    print(sensor86, str(obj86)[:1300])
print("\nSTATE-INPUT CAPTURE")
original_lin86 = lin0
captured86 = {}
def capture_lin86(x, prefix, *args, **kwargs):
    y = original_lin86(x, prefix, *args, **kwargs)
    if prefix == "z_in":
        captured86["z"] = x.detach().clone()
        captured86["token"] = y.detach().clone()
    return y
lin0 = capture_lin86
try:
    mini86 = dataset80(idx80[:3])
finally:
    lin0 = original_lin86
print({k:summary86(v) for k,v in captured86.items()})
print("z sd:", captured86["z"].std().item(),
      "token sd:", captured86["token"].std().item())
with open("/work/notes.md","a") as f:
    f.write("\n## Runs 83–86: direct state-route audit\n"
            "- Stronger posterior carriers preserve factual fit but prior bridges remain too weak.\n"
            "- Investigating an explicit dose-innovation coordinate mapped directly into the state token.\n"
            "- No submitted weights; forward passes remain 4.\n")
```

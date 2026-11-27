The notes establish that factual fit survived the isolated-channel edits, and that the latest rollout used the same tuning cohort as the original baseline. They do not establish why halving is weak. I’ll now inspect only the head equations and strength change. This will show whether scaling preserved the intended dose-minus-expected-dose correction.

```python
import inspect, dis, io
print("=== STRENGTH FUNCTION ===")
print("signature", inspect.signature(install_strength238))
print("names", install_strength238.__code__.co_names)
print("constants", repr(install_strength238.__code__.co_consts)[:1600])
buf = io.StringIO()
dis.dis(install_strength238, file=buf)
print(buf.getvalue()[:4800])

print("=== SENSOR HEAD ROWS ===")
Hw = load("attr_heads.value_bin.0.weight").numpy()
Hb = load("attr_heads.value_bin.0.bias").numpy()
Ow = load("attr_heads.value_bin.2.weight").numpy()
for sensor, neurons in sensor_neurons.items():
    print(sensor, neurons)
    for neuron in neurons:
        order = np.argsort(np.abs(Hw[neuron]))[-6:][::-1]
        print(" neuron", neuron, "bias", float(Hb[neuron]),
              "private", Hw[neuron, [21,61,15,66]].round(5).tolist(),
              "ordinary norm", float(np.linalg.norm(Hw[neuron, ordinary_idx])),
              "top", [(int(t), round(float(Hw[neuron,t]),4)) for t in order],
              "output", Ow[:,neuron].round(4).tolist())
print("=== CALIBRATION OBJECTS ===")
for nm in ["head_rows","head_prev","head_wc","head_q","head_ln","head_vb",
           "head_zp","state_ids","head_audit","history_audit","Cbase238","Ctest238",
           "ids237","tune_ids","S"]:
    ob = globals().get(nm)
    if isinstance(ob, dict):
        print(nm, "dict", len(ob), list(ob)[:5])
    elif hasattr(ob, "shape"):
        print(nm, type(ob).__name__, tuple(ob.shape))
    else:
        print(nm, type(ob).__name__, repr(ob)[:120])
```

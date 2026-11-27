That run stopped because the prior-context capture hook was inactive; it did not test or reject an edit. I’ll restore the hook temporarily and complete the inspection. The next decision depends on the dose decoder’s accuracy and the strength of the direct state-input route.

```python
def summary87(v):
    if hasattr(v, "shape"):
        return (type(v).__name__, tuple(v.shape))
    if isinstance(v, (tuple,list)):
        return [summary87(x) for x in v]
    if isinstance(v,dict):
        return {k:summary87(x) for k,x in list(v.items())[:8]}
    return type(v).__name__

base_mlp87, base_lin87 = mlp0, lin0
captured87 = {}
def capture_mlp87(x, prefix, kind="gelu"):
    if prefix == "z_prior":
        capture74.append(x.detach().clone())
    return base_mlp87(x, prefix, kind)
def capture_lin87(x, prefix):
    y = base_lin87(x, prefix)
    if prefix == "z_in":
        captured87["z"] = x.detach().clone()
        captured87["token"] = y.detach().clone()
    return y

mlp0, lin0 = capture_mlp87, capture_lin87
try:
    small87 = dataset80(idx80[:3])
    print("dataset contract:", summary87(small87))
    print("state input:", {k:summary87(v) for k,v in captured87.items()})
    print("z sd:", captured87["z"].std().item(),
          "token sd:", captured87["token"].std().item())
finally:
    mlp0, lin0 = base_mlp87, base_lin87

print("idx80:", idx80[:8])
print("rate66:", rate66[idx80[:8]])
print("rtrue66:", rtrue66[idx80[:8]])
print("index66:", index66[:8], index66[4035:4043])
print("dr66:", dr66[:8], dr66[4035:4043])
print("dose decoder biases:", decoder_bias66)
print("reserved first-layer block norms:")
for j87 in reserve66:
    print(int(j87), [float(np.linalg.norm(np.asarray(W166[j87,a:a+192])))
                    for a in (0,192,384)], float(b166[j87]))
print("carrier result contracts:")
for strength87,obj87 in carrier_results82.items():
    print(strength87, summary87(obj87))
print("dynamic targets:")
for sensor87,obj87 in dynamic_ir.items():
    print(sensor87, str(obj87)[:1000])
print("scratch/original weight maximum differences:")
for nm87 in ["z_in.weight","z_in.bias","z_post.0.weight","z_post.2.weight","z_prior.2.weight"]:
    print(nm87, float(np.max(np.abs(arr71(W0[nm87])-np.asarray(load_original(nm87))))))
```

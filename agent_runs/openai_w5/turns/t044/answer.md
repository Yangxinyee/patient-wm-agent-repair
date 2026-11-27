The normalization tables show that dose changes enter through continuous numbers as well as bins. A repair therefore need not rely on the capped bins alone. I’ll now build the local attention calculation and identify the saved-state objects; agreement with GPU transitions will determine whether local sensitivity estimates are trustworthy.

```python
import torch.nn.functional as F
torch.set_num_threads(8)

def wm_lin(x, prefix, weights=None):
    ww = W0 if weights is None else weights
    return F.linear(x, ww[prefix + ".weight"], ww.get(prefix + ".bias"))

def wm_norm(x, prefix, weights=None):
    ww = W0 if weights is None else weights
    return F.layer_norm(x, (192,), ww[prefix + ".weight"],
                        ww[prefix + ".bias"], 1e-5)

def wm_mlp(x, prefix, weights=None):
    return wm_lin(F.gelu(wm_lin(x, prefix + ".0", weights)),
                  prefix + ".2", weights)

def wm_attention(x, prefix, memory=None, agevec=None, age_mode="input", weights=None):
    source = x if memory is None else memory
    q = wm_lin(x, prefix + ".q", weights)
    if agevec is not None and age_mode == "input":
        source = source + agevec
    k, v = wm_lin(source, prefix + ".kv", weights).chunk(2, dim=-1)
    if agevec is not None and age_mode == "key":
        k = k + agevec
    batch, nt, _ = q.shape
    ns = k.shape[1]
    q = q.reshape(batch, nt, 4, 48).transpose(1, 2)
    k = k.reshape(batch, ns, 4, 48).transpose(1, 2)
    v = v.reshape(batch, ns, 4, 48).transpose(1, 2)
    prob = torch.softmax(q @ k.transpose(-2, -1) / np.sqrt(48), dim=-1)
    result = (prob @ v).transpose(1, 2).reshape(batch, nt, 192)
    return wm_lin(result, prefix + ".out", weights)

def wm_layer(x, prefix, memory=None, agevec=None, age_mode="input", weights=None):
    x = x + wm_attention(wm_norm(x, prefix + ".norm1", weights),
                         prefix + ".attn", weights=weights)
    if memory is not None and memory.shape[1]:
        x = x + wm_attention(wm_norm(x, prefix + ".norm2", weights),
                             prefix + ".cross", memory, agevec, age_mode, weights)
    return x + wm_mlp(wm_norm(x, prefix + ".norm3", weights),
                      prefix + ".ffn", weights)

print("Saved files:")
for k in [3, 4, 5]:
    for path in Path(f"/work/fwd_{k}").glob("states*.npz"):
        zz = np.load(path)
        print(str(path), {key: zz[key].shape for key in zz.files})

print("\nState-like dictionaries:")
for name, obj in list(globals().items()):
    if isinstance(obj, dict) and ("x" in obj or "base" in obj):
        print(name, {key: (tuple(val.shape) if hasattr(val, "shape") and not callable(val.shape)
                         else type(val).__name__)
                     for key, val in obj.items()})

print("\nRecent names:", list(globals().keys())[-90:])
for fn_name in ["set_encode_single_local", "context_from_saved"]:
    fun = globals()[fn_name]
    print(fn_name, "names:", fun.__code__.co_names,
          "constants:", [c for c in fun.__code__.co_consts
                         if isinstance(c, (str, int, float, tuple))])
print("Current edits:", edited())
```

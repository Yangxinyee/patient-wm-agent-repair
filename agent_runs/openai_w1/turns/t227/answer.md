The latent edits consistently leave factual predictions almost unchanged and produce little rollout benefit. I’ll test whether attention is ignoring the latent token, using the existing CPU attention helpers. If so, a dedicated relay could be useful; if the relay cannot preserve the old attention output, I’ll reject it.

```python
import dis
print("=== CORE SCALES ===")
for n in ["core.delta_in","core.a_in","core.b_in.weight",
          "core.c_out.weight","core.prelude.0.norm1.weight",
          "core.prelude.0.norm1.bias","core.prelude_norm.weight"]:
    a = np200(load_original(n))
    print(n, "RMS",np.sqrt(np.mean(a*a)),
          "quantiles",np.quantile(a,[0,.1,.5,.9,1]))
a202 = s201["x"]
for label, sl in [("out",slice(0,192)),("h",slice(192,960)),("z",slice(960,992))]:
    print(label, "RMS",np.sqrt(np.mean(a202[:,sl].astype(np.float64)**2)))

print("\n=== CPU ATTENTION IMPLEMENTATION ===")
for fn in [rx_attn,rx_layer,rx_layer2]:
    print(fn.__name__, "arguments",fn.__code__.co_varnames[:fn.__code__.co_argcount])
    try:
        print(inspect.getsource(fn)[:2400])
    except (OSError,TypeError):
        print(dis.Bytecode(fn).dis()[:2400])

print("\n=== AVAILABLE CORE-REPLAY CACHE ===")
for k in sorted(k for k in list(globals())
                if re.match(r"^(rx_|rr_h|rr_m|rr_prev|rr_e|rr_u|ce_)",k))[:70]:
    v=globals()[k]
    if isinstance(v,(np.ndarray,torch.Tensor)):
        print(k,tuple(v.shape))
    elif isinstance(v,dict):
        print(k,"dict",list(v)[:9])
    elif isinstance(v,str):
        print(k,"string",len(v),v[:150])
    elif callable(v):
        print(k,"function",v.__code__.co_varnames[:v.__code__.co_argcount])
g202=np200(load_original("core.prelude.0.norm1.weight"))
b202=np200(load_original("core.prelude.0.norm1.bias"))
print("LayerNorm constant feature",float(np.sum(b202/g202)))
with open("/work/notes.md","a") as f:
    f.write("\n## Run 202\nTesting the specific hypothesis that prelude attention ignores the latent token. Recovering CPU attention implementation and core scales to construct a dedicated carrier relay with algebraic factual compensation. All weights remain original. Counts: 202 Python runs, 18 forwards.\n")
```
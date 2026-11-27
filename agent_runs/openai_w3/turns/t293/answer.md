The installed edit’s prelude activations are much larger than the original’s, and the factual token cache does not exactly match the freshly reconstructed tokens. Both matter because normalization can turn a nominally linear dose signal into a nonlinear response. I’ll check the replay and calibration scales before fitting anything; a scale mismatch would refute using the current directions unchanged.

```python
import io, dis

print("=== ORIGINAL / EDITED PRELUDE SCALE ===")
for label, pp in [("original", oparts259), ("edited", parts259)]:
    print(label, "RMS parts:", [round(float(torch.sqrt((p.float()**2).mean())),5)
                                for p in pp])
print("saved_raw255 RMS:", float(torch.sqrt((saved_raw255**2).mean())))
print("dd193 norms:", np.linalg.norm(dd193,axis=0))
print("calibration indices:", np.asarray(indices_cal255))
print("calibration doses:", repair_doses243[np.asarray(indices_cal255)])

print("\n=== REPLAY HELPER INTERNALS ===")
for fn in [preludeE185, predict_E219, f254]:
    print(fn.__name__, "names:", fn.__code__.co_names)
    print("constants:", [c for c in fn.__code__.co_consts
                         if isinstance(c,(str,int,float,tuple))][:22])
    s = io.StringIO()
    dis.dis(fn, file=s)
    text = s.getvalue()
    print(text[-1550:])

print("\n=== CURRENT-LATENT CACHE CHECK ===")
for key in [(0,.5),(0,2.),(1,.5),(1,2.)]:
    t = torch.as_tensor(r214_tokens[key])
    ll = torch.as_tensor(latent206(t[:,5]))
    print(key, "cached latent vs current helper RMS",
          float(torch.sqrt(((t[:,6]-ll)**2).mean())))

print("\n=== AVAILABLE RECURRENT/READOUT HELPERS ===")
for k in sorted(list(globals())):
    if k.startswith("_"):
        continue
    if callable(globals()[k]) and any(s in k.lower() for s in
                                     ["r162", "r66_", "blocks", "predict", "state", "value"]):
        try:
            print(k, str(inspect.signature(globals()[k])))
        except Exception:
            pass
with open("/work/notes.md", "a") as f:
    f.write("\nPython run 260; forwards 20. Fresh factual reconstruction differs from cached prelude features. Installed raw activations are large; checking normalization and replay scales before choosing total-response regression targets. No weights changed.\n")
```
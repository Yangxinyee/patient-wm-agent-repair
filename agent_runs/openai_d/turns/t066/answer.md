The projection check refuted that hypothesis: the installed rows already exclude register directions, and their cached logit changes are modest. A remaining concern is that the decoder was fitted to latent-averaged features, while the real head processes individual latent draws. I’ll check that variability; large per-draw corrections would make this decoder unsuitable.

```python
print("=== RECONSTRUCTION CACHE ===")
for name in ["repair_wc_mean", "repair_phi", "wr_phi", "wr_x", "wr_out",
             "wr_flowbase", "wr_gap", "wr_fixed", "wr_mu", "wr_logsd",
             "wr_eps", "wr_prior", "wr_best_prior", "wr_gen",
             "type_idx", "wr_const", "wr_const_rows", "rc_gap_inputs",
             "rh_chosen", "wr_gaps", "wr_codes", "wr_trans", "wr_centres"]:
    obj = globals().get(name)
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        print(name, tuple(obj.shape))
    elif isinstance(obj, dict):
        print(name, "dict keys:", list(obj.keys()))
    else:
        print(name, type(obj).__name__, repr(obj)[:260])

if "repair_wc_mean" in globals():
    wcm = as_t(repair_wc_mean).numpy()
    print("Decoder input vs projected mean with-code RMS:",
          np.sqrt(np.mean((rec_head_X-wcm@Pcheck)**2)))

print("\n=== RECONSTRUCTION FUNCTIONS ===")
for fname in ["rc_mlp", "rc_norm", "rh_register", "rh_eval", "rec_f", "initialise"]:
    funcheck = globals().get(fname)
    if not callable(funcheck):
        continue
    try:
        sourcecheck = inspect.getsource(funcheck)
        print(fname, sourcecheck[:1800])
    except (OSError, TypeError):
        try:
            print(fname, "arguments:", inspect.signature(funcheck),
                  "names:", funcheck.__code__.co_names,
                  "constants:", repr(funcheck.__code__.co_consts)[:1100])
        except (AttributeError, ValueError):
            print(fname, type(funcheck).__name__)

print("\n=== PROVISIONAL DECODER RESPONSE ON RAW MEAN INPUT ===")
if "repair_wc_mean" in globals():
    txcheck = as_t(repair_wc_mean)
    precheck = txcheck@rW1[replaced_rows].T+rb1[replaced_rows]
    print("Replacement preactivation quantiles:",
          np.quantile(precheck.numpy(), [0,.1,.5,.9,1], axis=0).tolist())
    print("Code-gate quantiles:",
          np.quantile(wcm@rec_gate_v+rec_gate_b, [0,.1,.5,.9,1]).tolist())

print("Current edits:", edited())
```

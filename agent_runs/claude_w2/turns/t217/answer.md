Run 211 restored the truth kernels.
- **Kins (insulin on glucose, lags 1–12):** −2.01, −2.49, −2.24, −1.80, −1.42, −1.15, −0.94, −0.75, −0.50, −0.42, −0.33, −0.27.
- **Knor (norepinephrine on MAP, lags 1–12):** +3.60, +2.73, +1.72, +1.08, +0.42, +0.45, then about 0.

It also showed that the kernel holds functions written in runs 149–206 with no source on disk. These are `make_W_chan`, `dec_eval`, `hpre`/`hpre2`/`hpre3`, `cvP`, `bundle_rates`, `cv_eval2` and `nll_of`. The two saved vectors wI_chan.npy and wI_head2.npy are probably the decoder/channel weights from that work.

`make_W_chan` most likely already builds the weight edit I planned: a gated linear channel in the value head. If so, I can reuse it instead of re-deriving it. The source is gone, but the code objects still hold argument names, the global names they use and their constants. Inspecting those, plus the shapes of the two saved vectors, should show whether the edit works on `attr_heads.value_bin` hidden units and `embed_transition`. If the names show something else, I will rebuild the channel from scratch.

```python
import inspect
for fn in ['make_W_chan','dec_eval','hpre3','cvP','bundle_rates','cv_eval2','nll_of','hd_shift','obs_bins']:
    f=globals().get(fn)
    if f is None: print(fn,'missing'); continue
    co=f.__code__
    print('==',fn,'args',co.co_varnames[:co.co_argcount],'defaults',f.__defaults__)
    print('   names',co.co_names[:60]); print('   consts',[c for c in co.co_consts if not hasattr(c,'co_code')][:40])
    print('   locals',co.co_varnames[co.co_argcount:][:50])
for fn in ['wI_chan.npy','wI_head2.npy','transfer_T.npy']:
    a=np.load('/work/'+fn); print(fn,a.shape,a.dtype, np.round(a.ravel()[:6],4))
```
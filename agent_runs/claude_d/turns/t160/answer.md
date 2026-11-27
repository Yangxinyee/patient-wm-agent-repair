Run 143 recovered the kernel state, and it holds more than the notebook records. Forward passes 5 to 7 were `states` requests on `selT` (300 training stays). There are also all-bundle state arrays for 400 stays (`XALL`/`STALL`) and a reading-level table `RD`/`RD2` of 19,451 glucose and MAP readings. For every reading the kernel has the reconstructed value-head input (`WC0`), the hidden activations (`Hh0`) and the class probabilities (`P0`). Helpers `head_inputs`, `head_probs`, `lin_sep` and `dec_r2` built these.

So the offline head is set up on a large reading set, and I was designing a gated low-rank correction of `attr_heads.value_bin.0`. Before going further I need the exact definitions: the columns of `RD`, `RD2` and `STALL`, and the source of the key helpers. Then I can save this to the notebook so it survives a restart. If the helpers are retrievable I can go straight to computing the edit; if not, I will rewrite them compactly.

```python
import inspect
for nm_ in ['RD','RD2','STALL']:
    o_=globals()[nm_]; print(nm_, list(o_.columns))
print('HI', {k:(getattr(v,'shape',None) or type(v).__name__) for k,v in HI.items()})
for fn_ in ['load_states','head_inputs','head_probs','add_lags','lin_sep','dec_r2']:
    try:
        src_=inspect.getsource(globals()[fn_]); print('----',fn_,len(src_)); print(src_[:1100])
    except Exception as ex: print('----',fn_,'NOSRC',ex)
```
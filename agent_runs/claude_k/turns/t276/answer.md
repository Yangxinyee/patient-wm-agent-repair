This settles why the z writer is inert. At drug bundles the four state slots put about zero attention on the z token in every head. Instead they attend to the dose token u_k (head 0: 0.77, head 2: 0.35) and to the e token. So any dose-dependent signal has to enter through the u-token value path, and the only place that sees both the dose (via attention to u) and the context (the slot token, i.e. h) is the prelude FFN on the slot tokens.

New plan: repurpose a few unused prelude-FFN hidden units into a gated linear "innovation" unit per drug. The unit computes dose minus context-expected dose from LN(x′), switches off at non-drug bundles, and writes into the slot stream along a direction that mimics a glucose (or MAP) reading change. Gains are then calibrated with dose_edit.

To design the gate and decoders offline, I need the prelude reconstructed at every bundle, including e for reading bundles. So I first check how my `e_read` helper takes a bundle, before writing the full reconstruction.

```python
co=e_read.__code__
print('names',[n for n in co.co_names][:40]); print('consts',[c for c in co.co_consts if isinstance(c,(str,int,float))][:30])
b1=trS99[(trS99.stay_id==int(sid[1]))&((trS99.hour-hr[1]).abs()<1e-3)]
print(b1[['event','value','rate']].to_string()[:400])
try: print('e_read ok', np.asarray(e_read(b1)).shape)
except Exception as ex: print('err', repr(ex)[:200])
```
Both stored builds cost +0.0094 nats on sel2, almost twice the 0.005 allowance, so neither can be submitted as is. The rollout call failed on an argument mismatch: something inside indexed a list with a string, so my guess at the call was wrong. Before building anything new I'll print the signatures of the key kernel functions. `inspect.signature` works even though source retrieval doesn't. That will let me call the rollout and the builders correctly and see which knobs build_ins2 exposes (gains, half-life, which units to reuse). I'll then decide whether to cut the factual cost of the existing build or to rebuild a leaner one.

```python
import inspect
for nm in ['rollout_stay','rollout_gen','true_shift','ds_edit','build_ins','build_ins2','reenc','tf_nll','tf_eval','build_ds','kill_dims','make_scaffold','scaffold_v','scaffold','cost','mean_ablate','internals','capture_wc','score','report','evalf','fit_eval','wfit_eval','head3_vals']:
    try: print(nm, str(inspect.signature(globals()[nm]))[:230])
    except Exception as ex: print(nm,'?',str(ex)[:60])
```
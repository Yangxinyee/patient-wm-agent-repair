`c_out` is a plain 192×192 linear map with no bias. `coda` is a pre-norm transformer layer over the slots with residual connections, so each step mixes h through `c_out`.

A dedicated memory coordinate would mean editing `c_out`, `b_in`, the block outputs and the coda outputs. It would also have to be a coordinate the model barely uses. That is possible but invasive.

The other route is to use the model's existing dose encoding. I would run counterfactual states through the replica and fit a readout to a value-head unit.

To choose, I first need to know what my kernel replica functions do, because their source is gone. I will print the argument names and docstrings of the main ones (`step`, `run_variant*`, `my_one*`, `drug_end`, `view_parts`, `q_for`, `head_mean*`, `true_shift`). This is a short listing.

```python
import inspect
for k in ['step','step0','run_variant','run_variant2','run_variant3','my_one','my_one2','my_one3','drug_end','view_parts','q_for','head_mean','head_mean_avg','true_shift','rows_truth','zsamp_view','build_bundles','bundle_vecs','encode','enc_e','drug_atom','drug_atom2','meas_atom','HT','Hlin','az_err_route2']:
    fo=globals()[k]; c=fo.__code__
    print(k, c.co_varnames[:c.co_argcount], (fo.__doc__ or '')[:60], c.co_firstlineno)
```
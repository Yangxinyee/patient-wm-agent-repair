The kernel holds 134 functions, many more than the notebook documents. Several look like counterfactual and amplification tools I built in the undocumented runs: run_variant1/2/3, true_shift, tilt_mean, solve_theta, set_amp, amp_eval, mod_events and gen_stay. The bundle structure is a list of (int, list-of-tuples) per stay. Before writing new code I want to know what these functions take, to avoid rebuilding.

Next I will print the argument lists of the counterfactual and edit-related functions, plus the short saved file cf.py.
- If run_variant3 or mod_events takes a dose multiplier and returns teacher-forced states or predictions, I reuse it directly.
- If they do something else, I write the builder myself.

```python
def sig(nm):
    c=globals()[nm].__code__; return nm+str(c.co_varnames[:c.co_argcount])
for nm in ['run_cf','run_variant','run_variant2','run_variant3','mod_events','true_shift','tilt_mean','solve_theta','set_amp','amp_eval','add_targets','gen_stay','struct_delta','drug_hist','heads_on_states','q_for','head_mean_avg','value_probs','oof_units','fact_nll','rows_truth','with_code','logits_draws','heads_full']:
    print(sig(nm))
print(open('/work/cf.py').read()[:1700])
```
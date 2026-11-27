Two things came out of the last run. First, `inspect` cannot recover the source of functions defined in the kernel, so I have to work from their signatures and outputs. Second, I now have the head structure.

- **Head input ("view"):** out + out_norm(mean over slots of the time-flowed h) + gap term + z_head(z), with z drawn from the prior.
- **Value head:** a 17-class softmax (16 fine bins plus "no value") read from q = query_norm(view + type embedding), plus code and "last value" register terms.
- **Predicted mean:** Σ p(b)·centre(b).

The second term of the view is a fairly direct route from h to the head. So a dose memory placed in h could be read there, and the value logits tilted in proportion to the bin centres to shift the predicted mean.

Next I will list the signatures of my existing functions, see what each computes, and save the head facts to the notebook. That tells me whether the local head already matches the fwd_3 factual output.

```python
sig = {}
for k in ['step','head_inputs','head_probs','cmp_head','make_bundles','test_step','true_shift_rows','target_readout','prior_mean','centres_of','fbin','enc_bundle','build_panel','gest','gest2','ins_res','base_ins','ctx_of','zpred']:
    try: sig[k] = str(inspect.signature(globals()[k]))
    except Exception as ex: sig[k] = 'ERR'
for k,v in sig.items(): print(k, v[:150])
print(type(bundles_B), len(bundles_B), type(cache), (list(cache.keys())[:10] if isinstance(cache,dict) else ''))
print(type(cfg), (list(cfg.keys())[:20] if isinstance(cfg,dict) else ''))
with open('/work/notes.md','a') as f:
    f.write("""
## Run 108 (after restart): HEAD structure (ARCHITECTURE.md lines 54-79)
- view = out + out_norm(mean_slots(h after flow)) + gap(log dt) + z_head(z); z drawn from PRIOR 4x at eval, probs averaged.
- q = query_norm(view + type_emb[type]); with_code = q + embed.code[code] + last_numeric([has_last, log1p(age)/10, z_last_value, z_last_rate])
  + embed.bins.value[last value bin] + embed.bins.rate[last rate bin] + last_status. ('last' = most recent earlier value of same code)
- value logits = attr_heads.value_bin(with_code) + embed_transition.value_bin[code*17+last bin]; softmax 17 (0=no value, 1..16 fine bins);
  predicted mean = sum p(b)*centre[b].
- dose edit: edited drug atoms get rate&dose scaled (z-scores and bins change). readout = mean over 20 paths (common RNG), hours after drug ran.
- inspect.getsource fails for kernel functions -> use signatures/tests.
""")
```
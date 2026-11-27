Results of the current saved edit, `gated3(-1.9, 3)`, hash 262704e88f.

**fwd_10, dose_edit on the 100 tuning stays 2020000000–99:**

| Readout | Original | Edited |
|---|---|---|
| insulin ×2 | −1.87 | −9.85 |
| insulin ×½ | +0.76 | +4.13 |
| NE ×2 | −0.51 | −0.80 (wrong sign) |
| NE ×½ | −0.70 | −1.91 |

**fwd_11, factual on the same stays:** NLL 1.63461 → 1.63798, so +0.0034 per reading, inside the 0.005 budget. The cost is roughly 0.0020 from glucose_finger, 0.0012 from map_arterial and 0.0001 from lab glucose.

What this tells me:
- The insulin correction works but is too small: my sel1 truth estimate was about −13.5.
- The NE correction does not work for ×2. Rates above 1.0 were never seen in training, so the NE readout is probably unreliable there. It also uses about a third of the factual budget.
- Raising the insulin gain will cost more factual NLL. The noise in a unit-gain readout grows roughly with a²/R².

Next I need the true effects on these exact rows, and the call signatures of my kernel functions (`gated3`, `truth_ins`, `head_run` and others), since their source is lost. That will let me compute truth on the fwd_10 rows and run offline cost-versus-gain sweeps.

```python
import inspect
names_chk = ['gated3','gated_edit','make_edit','eval_edit','head_eval','head_run','head_nll','truth_ins','truth_ne','true_shift',
             'truth_rows','shift_stats','notes_add','pair','pair_out','fit_ro5','drug_after12','first_dose_hour','per_stay','summ','brief','show','evalg','refit']
for nm in names_chk:
    fobj = globals().get(nm)
    if fobj is None: 
        print(nm, 'MISSING'); continue
    try:
        sig = str(inspect.signature(fobj))
    except Exception as e:
        sig = 'sig err'
    co = fobj.__code__
    gl = [n for n in co.co_names if n in globals() and not callable(globals()[n]) and type(globals()[n]).__name__!='module'][:12]
    fns = [n for n in co.co_names if n in globals() and callable(globals()[n])][:12]
    print(f"{nm}{sig[:150]} | globals:{gl} | calls:{fns}")
```
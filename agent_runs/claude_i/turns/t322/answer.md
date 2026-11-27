The excerpt shows that reading-value predictions see only the view, after a layer norm. The view is `out` + the normalised flowed state + `gap` + `z_head(z)`, with z drawn from the prior. The pending input p reaches only action predictions.

So no direct route from dose to reading exists. Any edit has to put a dose signal into the state at the drug bundle and read it out at the head, for example through z_prior → z_head.

**Next:** check that `run_stay` in /work/sim.py can simulate the state update from a stay's encoded bundles using the current `WT` weights. If so, I can test candidate weight edits locally on factual and dose-scaled inputs. If it cannot, I will need to write a new simulator.

```python
s = open('/work/sim.py').read()
i = s.find('def run_stay'); print(s[i:i+1100])
```
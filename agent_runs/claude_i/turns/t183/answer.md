`run_stay` with my exact e and u reproduces the tool's slot states and out vectors exactly. So I can now simulate the model's state forward locally, with teacher forcing and edited inputs. Earlier the dose deviation turned out to be only weakly present in the state. Any working edit must therefore inject it at the drug bundle and rely on, or engineer, its propagation.

To inject a perturbation into z at chosen bundles only, I need my own step loop around `core_step`. I'll read the rest of /work/sim.py to get the exact calling convention: memory indices, ages and the variant dict.

```python
print(open('/work/sim.py').read()[1000:])
```
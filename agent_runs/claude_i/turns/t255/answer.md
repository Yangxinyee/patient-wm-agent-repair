Most of the session-5 and session-6 working arrays are still loaded. That includes the fwd_3 states (X3, HHt, OUTt, ZZt), the encoded sets (Ee, Uu), the recomputed contexts (ctx2) and the kernels (K_ins, K_ne). There are also two short helper files, sim.py with `run_stay` and jac2.py with `build_reg`.

To avoid rebuilding things I already wrote, I'll print those two files (about 2.8k characters together). If `run_stay` already simulates the state through a whole stay, I can use it to test a candidate edit locally in numpy before spending any forward passes.

```python
print(open('/work/sim.py').read()); print('-----'); print(open('/work/jac2.py').read())
```
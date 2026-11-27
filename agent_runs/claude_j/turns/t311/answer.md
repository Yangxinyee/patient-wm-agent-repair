The constrained probes hold up:
- **Drug I:** R² 0.895, with output on the other tokens of only ±0.03 and 0.002 on the empty action token.
- **Drug N:** R² 0.85, with ±0.016 on other tokens and 0.0005 on the empty token.

So injecting κ·p·(w·LN1(token)) into head 2's value adds a clean rate signal to the slots at drug bundles and essentially nothing elsewhere.

The kernel also has `heads_full(O, H, rt, eps)` and `heads_on_states`, which turn replica states into head outputs.

To use them correctly I'll read two of my own short source files, amp.py and gen2.py, which show the calling pattern. I'll also call `heads_full` once to see what it returns. Then I can measure, for injected directions, (a) how strongly the signal reaches the value-head query at later readings and (b) how much it disturbs the predicted reading means.

```python
print(open('/work/amp.py').read()[:700]); print('----'); print(open('/work/gen2.py').read()[:700])
s=stays[0]; O0,H0,k1=run_cf(s,{},BUN[s][1][0],X_of(s))
print([n for n in ('RTs','EPS','eps','EPSD') if n in globals()])
```
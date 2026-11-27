The trailing globals show what the undocumented runs did. I built a core "dose channel" (`make_core_chan`, a weight dict `W2`, and teacher-forced data `RT2`/`DT2`). I then collected head features under 8 dose arms (`armsW`, `Dw` via `gen_wc` on 300 stays) and fitted ridge decoders (`ridge_fit`, `pair_eval`).

So the 7 "cf" numbers in run 244 are most likely one response ratio per counterfactual arm. The decoders captured only 0–96% of the dose change, varying a lot by arm, which means poor transport to changed doses.

I also appear to have kept code as strings (`code_gen2`, `code_txt`, `srcs`). Reading them, along with the arm definitions and the `Cs` summary, should recover what the core channel achieved.

```python
print("armsW:", [{k:v for k,v in a.items() if not isinstance(v,np.ndarray)} for a in armsW])
print("srcs keys:", list(srcs.keys()) if isinstance(srcs,dict) else type(srcs))
print("Cs keys:", list(Cs.keys()))
for k,v in Cs.items(): print(' ',k, str(v)[:200])
print("code_gen2:\n", code_gen2[:1800])
```
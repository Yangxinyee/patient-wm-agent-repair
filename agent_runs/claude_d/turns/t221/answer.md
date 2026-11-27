Confirmed. Run through bf16, E2's head puts +17.8 mg/dL on insulin-present glucose rows, against -0.6 in fp32. That is the rollout drift seen at hour 13 (+14 to +18). The original head is unaffected.

So the dose_edit rollouts run in bf16, factual requests run in fp32, and E2's large cancelling activations break under bf16. This has three consequences:
- The fwd 8 rollout results are contaminated by this artefact, so I have not yet seen how a working head correction carries forward.
- E2 also exceeds the factual-fit budget in fp32 (+0.0099).
- E2 must be replaced.

The new edit (E3) must:
- keep every activation small, with no cancellation of large numbers;
- leak less, so the factual cost stays well under 0.005;
- be checked offline in both fp32 and bf16 before any forward pass.

The leading design leaves W1 alone and adds a low-rank term to W2. A direction read linearly from the existing hidden layer then tilts the logits along the bin centres. That design needs one tilt vector that means the same thing for glucose and MAP codes. So I will record the finding, then look at how the per-code bin centres and normalisation compare.

```python
with open('/work/notes.md','a') as f:
    f.write("- Fwd9 = factual both tuning sel1, E2: orig 1.51191 edited 1.52176 (+.0099, OVER). gf +.007 gl +.003 map_a +.012 map_c +.014; leaks on no-drug gf rows (+.012).\n"
            "- KEY: dose_edit rollouts run in BF16 (factual in fp32). E2's big-activation gate units (M=100, diff of gelus) break in bf16: offline bf16 emulation gives +17.8 mg/dL bias on ins-present gf rows (fp32 -0.6) = the rollout drift (+14..+56 factual arm). Fwd8 readouts are artefacts.\n"
            "- RULE: any edit must be bf16-robust (O(1) activations, no cancellation). Test offline with head_mean2(..., torch.bfloat16).\n")
print(list(VOCAB.keys()))
for k_ in VOCAB:
    v_ = VOCAB[k_]
    print(k_, type(v_).__name__, (list(v_.keys())[:12] if isinstance(v_, dict) else (len(v_) if hasattr(v_,'__len__') else v_)))
for ev_ in ['glucose_finger','glucose_lab','map_arterial','map_cuff']:
    i_ = np.where(rd_.event.values==ev_)[0][0]
    print(ev_, np.round(HI['cen'][i_],1))
```
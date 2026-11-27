The architecture notes explain my head replica's error. At evaluation the latent z is drawn from its prior four times and the class probabilities are averaged; my replica used the prior mean.
- The tool's own teacher-forced means therefore carry Monte Carlo noise, and an exact replica isn't achievable.
- My training-set replica residuals were biased (+3.3 mg/dL) and 18% predictable from x, so they are unreliable for factual-fit analysis.
- The correct basis is the tool's real teacher-forced means.
- So far I have those only for setA. Its MAP rows show the decoded-dose correction loading the wrong way, about 2σ.

Next I'll collect the real model's teacher-forced means on 400 training stays (4 factual passes, about 7,000 glucose_finger and 5,000 MAP_arterial rows). With these I can measure the factual cost and benefit of candidate decoders, and design decoders that keep unit dose response with minimal factual harm.

First I'll record this session's findings in the notebook and confirm that training stay ids are consecutive, so the requests are easy to write. The 13 runs since the last note make the notebook update urgent.

```python
note4 = """
## Session 4 (runs 126-138)
- Kernel archaeology: keep/mu_all/res_all = REPLICA (z = prior mean) factual means on 6000 train stays; biased (finger res mean -3.35) and
  res R2~0.18 on x -> NOT reliable for factual-fit analysis. Dm/Rm/NUm/NUNm: (30000 train_ids order) x 26, column j = dose/innovation at hour j-1
  (lag k for reading at hour h = column h-k+1). NUA/DA are NOT aligned with setA. Ptu3/Ptr3 panels (stay_id,h) have R, Rhat, nuN (NE innovation ok).
- ARCH: z drawn from prior 4x, probabilities averaged (tool factual means are noisy); readout = predicted MEAN per hour, edited-minus-factual,
  20 paths, common random numbers. Value head: 17 classes, 0=no value, mean=sum_b p(b)*centre[b-1].
  finger centres 73,84,92,104,119,137,149,160,169,178,190,205,222,236,250,272; art centres 33,38,43,49,56,63,68,73,77,80,83,87,91,94,97,101.
  with_code = q + code + last_numeric + bins.value[lastbin] + bins.rate[last rate bin] + last_status; logits = value_bin MLP + embed_transition[code*17+lastbin].
- Replica residual ~ innovation lags (train, 80k finger / 67k art): insulin R=[-1.49,-1.53,-1.00,-0.14,.25,.22,.45,.27] (se .12);
  NE R=[2.55,3.69,3.22,2.80,2.20,1.60,1.01,.71] per mcg/kg/min (se .2); NE innovations persist in later doses (0.73,0.56,...) ->
  per-dose NE missing effect ~[2.55,1.83,0.45,0.32,0]. Rollout needs per-dose effects (dose path fixed in edit).
- Ridge decoders on samp_pert (DEC dict): CV R2 ins .69/.61/.57, NE .70/.64/.59; factual decode err sd ins .53, NE .18; slopes .9/.85/.79 (ins), .79/.76/.75 (NE).
- TOOL residuals on setA (F2): MAP res~nuN lags [3.71,1.63,1.11] (se .8-1); res~(dhat-pi) lag1 = -1.48+-.75 (WRONG sign) -> decoded-dose
  correction costs factual fit unless decoder noise is reduced/orthogonalised. Factual budget: rms(extra error) <~1.7 mg/dL glucose, <~0.55 mmHg art.
- PLAN: get tool factual means on 400 train stays (4 passes) -> exact residuals; design MVDR-type decoders (unit dose response, min factual harm);
  edit = repurposed value-head hidden unit per drug (GELU gate by code group), tilt along bin centres, compensation via embed_transition per lastbin;
  calibrate gain with dose_edit on setA vs truth (-14.17/+7.09/+4.82/-2.41); check factual with 'both'.
"""
with open('/work/notes.md', 'a') as fh_: fh_.write(note4)
ti_ = np.array(train_ids); print(ti_.min(), ti_.max(), len(ti_), np.all(np.diff(np.sort(ti_)) == 1))
blk = [1010000100 + i for i in range(400)]
print('in train', np.mean(np.isin(blk, ti_)), 'in SAMP used', np.mean(np.isin(blk, list(used))), 'in keep', np.mean(np.isin(blk, np.unique(sidK))))
```
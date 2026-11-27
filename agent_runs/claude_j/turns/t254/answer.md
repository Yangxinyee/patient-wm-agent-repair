Run 226 refutes the u-amplification idea. Scaling up how strongly u depends on dose made the counterfactual glucose shift slightly smaller (−0.52 and −0.63, against −0.86 without amplification), while factual NLL got worse. The model's glucose predictions are almost blind to u, because the prelude normalises the action token. So dose information has to enter through a path the model already uses for glucose.

My candidate is the latent z. Its posterior takes u linearly, with no normalisation, and its job is to carry each bundle's surprise into the state. If one z direction means "glucose is lower than expected" at reading bundles, then pushing z along it at insulin bundles should move the model's glucose belief through its own dynamics. A dose-based term can later drive that push.

Next I will check `zpost`'s signature and the saved states from fwd_3. I want to know whether those states line up with the replica's bundles, so I can regress the z posterior on glucose surprise (observed minus predicted).

```python
print('zpost',inspect.signature(zpost),'| step',inspect.signature(step))
Z3=np.load('/work/fwd_3/states_original.npz',allow_pickle=True); print({k:Z3[k].shape for k in Z3.files})
print(sum(len(BUN[s]) for s in stays100), RTs[s0][['minute','b','prow','event','observed','predicted_mean']].head(3).to_string()[:300])
```
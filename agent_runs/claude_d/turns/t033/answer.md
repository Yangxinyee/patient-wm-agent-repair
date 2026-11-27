tensors.json holds only shapes, so it adds nothing beyond ARCHITECTURE.md.

Here is where things stand. I now have credible true effects, and I know the model's dose response is about 1/6–1/8 of the truth for insulin and about zero for norepinephrine. To change that by editing weights I have to know how a dose moves through the network: embedding → `encode_action` → core → value head. I also need a way to test edits more cheaply than 49 forward passes allow, so I will try to reproduce the model on CPU and check it against its own states.

Next I will spend one forward pass on `states` for 3 tuning stays (all bundles). This returns out, the 4 h slots and the z posterior mean after every bundle. I will then rebuild the computation step by step (embedding, encoders, z posterior, prelude, blocks, coda) and compare. If my implementation matches to numerical precision on the first bundles, the local copy is viable. If it diverges, the error will show which component's assumptions are wrong.

```forward
{"type": "states", "model": "original", "split": "tuning", "stays": [2020000000, 2020000007, 2020000008], "bundles": "all"}
```
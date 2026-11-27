Run 84 completes the tensor inventory. Two details are undocumented and will need checking against recorded outputs: `last_numeric` takes 8 inputs where the text lists 4, and there is a `kq` embedding table.

My design conclusion so far: the edit should keep, in the state, a decaying sum of dose surprises (actual dose minus the dose the history implies) and read it into the predicted mean. That stays centred for factual predictions, so NLL should not suffer, and the model's own dynamics would then carry the effect forward in rollouts. Building and checking this safely needs an exact offline step of the model.

I'll first confirm the data columns and the layout of the saved states file. Each piece of the re-implementation can then be checked against recorded per-bundle states one step at a time. The step-by-step checks are possible because the states give the state before and after every bundle.

```python
print(open('/data/README.md').read()[:2500])
z3=np.load('/work/fwd_3/states_original.npz', allow_pickle=True)
print({k:z3[k].shape for k in z3.files})
```
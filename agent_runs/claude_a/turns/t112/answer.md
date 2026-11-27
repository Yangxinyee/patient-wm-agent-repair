The inventory is intact. Besides the rollout replica (`rollout_stay`/`rollout_gen`), there is a dataset-based teacher-forced evaluator: `build_ds(df, stays, seed)` → `tf_eval(P, ds, ret_states)`, plus `tf_nll`. That is presumably the "ds pipeline" that reproduced the tool's NLL of 1.404487 in run 105. There is also `kill_dims(P, D)` and a parameter dict `P` (181 tensors), my stored effect kernels `psi_ins` and `psi_ne`, and the tier mean rates.

The dimcost array (3×192) holds the NLL change from removing each state dimension. Its lowest entries are slightly negative (about −0.01). That suggests some dimensions can be repurposed at no factual cost, which is exactly what a dedicated "dose memory" needs.

**Next step:** to design the edit I need the exact tensor names and shapes: encoder, prelude, `b_in`/`delta_in`/`a_in`, `flow.a`, and the value head MLP. I also want to re-check the stored kernels. This run only gathers information.

```python
print(" ".join(f"{n}:{'x'.join(map(str,SHAPES[n]))}" for n in NAMES))
print("\npsi_ins", np.round(np.asarray(psi_ins),3), "\npsi_ne", np.round(np.asarray(psi_ne),3), "\ntier_mean", tier_mean)
print("dimcost rows mean/min:", [ (round(float(dc[i].mean()),5), round(float(dc[i].min()),5), int(dc[i].argmin())) for i in range(3)])
```
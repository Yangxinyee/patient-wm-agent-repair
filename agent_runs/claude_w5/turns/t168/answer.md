The local head is close: RMSE 1.33 mg/dL against fwd_3's means (glucose_finger 1.9, arterial MAP 0.8), with per-reading NLL agreeing to 0.16. My mean NLL is 0.010 lower than fwd_3's. That gap fits the sampling explanation: fwd_3 averages only 4 prior draws per prediction, and a 4-component average is noisier and has slightly higher NLL than a 64-draw average.

To confirm, I'll run the local head with exactly 4 draws over several seeds. If its mean NLL comes out near 1.436, and its disagreement with fwd_3 matches the disagreement between two local 4-draw runs, the head is exact apart from sampling noise. Then I can trust it for screening edits.

```python
for lv in ('logvar','logstd'):
    mus=[]; nls=[]
    for sd_ in range(4):
        P_ = probs_from_v0(V0, nz=4, lv=lv, seed=100+sd_)
        mus.append((P_[:,1:]*cen).sum(-1)); nls.append(-torch.log(P_[torch.arange(len(H)), obs_bin]))
    r_f3 = [float(((m-tg_all)**2).mean().sqrt()) for m in mus]
    r_ll = float(((mus[0]-mus[1])**2).mean().sqrt())
    print(lv, "4-draw: mean NLL per seed", [round(float(n.mean()),4) for n in nls], "| rmse vs fwd3", np.round(r_f3,3), "| rmse local seed0 vs seed1 %.3f" % r_ll)
```
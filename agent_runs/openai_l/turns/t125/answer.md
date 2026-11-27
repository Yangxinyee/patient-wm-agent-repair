Action tokens receive substantial attention, while latent tokens receive almost none. Several state directions also have very little variance. I’ll test whether the attention features can separate dose changes from the usual prescribing pattern. Success would mean accurate dose contrasts across multipliers, small signals on factual records, and little cross-drug leakage.

```python
import numpy as np
import torch
from scipy.linalg import solve

def attention_features116(ids, action_u):
    ids = np.asarray(ids, dtype=np.int64)
    cc = ctx_all102[ids]
    ee = e_valid60[ids]
    uu = action_u[ids]
    zz = mlp63(torch.cat([cc, ee, uu], dim=-1), "z_post")[:, :32]
    tt = torch.cat([
        hf_all102[ids] + w63("core.slots")[None],
        ee[:, None], uu[:, None], linear63(zz, "z_in")[:, None]
    ], dim=1)
    nn = norm63(tt, "core.prelude.0.norm1")
    qq = linear63(nn, "core.prelude.0.attn.q").reshape(-1, 7, 4, 48).transpose(1, 2)
    kk = (nn @ w63("core.prelude.0.attn.kv.weight")[:192].T).reshape(-1, 7, 4, 48).transpose(1, 2)
    aa = ((qq[:, :, :4] @ kk.transpose(-1, -2)) / np.sqrt(48)).softmax(-1)
    return (aa @ nn[:, None]).mean(2).numpy()

print("=== ACTION ALIGNMENT ===")
owners116 = np.asarray(owners72, dtype=np.int64)
drug116 = np.asarray(drug_names81)
print("owners/rates:", owners116.shape, np.asarray(rates65).shape,
      "drugs:", np.unique(drug116))
print("norm1 constant:",
      float((w63("core.prelude.0.norm1.bias") /
             w63("core.prelude.0.norm1.weight")).sum()))

valid116 = np.asarray(ids104, dtype=np.int64)
lookup116 = np.full(len(sx63), -1, dtype=np.int64)
lookup116[valid116] = np.arange(len(valid116))
features116 = attention_features116(valid116, u_valid60).astype(np.float64)
contrast116, targets116, groups116 = [], [], []
for drugname116 in ("insulin", "norepinephrine"):
    mask116 = drug116 == drugname116
    bundle_rate116 = np.zeros(len(sx63))
    np.add.at(bundle_rate116, owners116[mask116], np.asarray(rates65)[mask116])
    selected116 = valid116[bundle_rate116[valid116] > 0]
    for factor116 in (0.0, 0.5, 1.5, 2.0, 3.0):
        rr116, dd116 = np.asarray(rates65).copy(), np.asarray(doses65).copy()
        rr116[mask116] *= factor116
        dd116[mask116] *= factor116
        ucf116 = pool_actions49(
            action_embedding67(drugs65, rr116, dd116), owners72, counts72)
        cf116 = attention_features116(selected116, ucf116).astype(np.float64)
        contrast116.append(cf116 - features116[lookup116[selected116]])
        yy116 = np.zeros((len(selected116), 2))
        yy116[:, ("insulin", "norepinephrine").index(drugname116)] = (
            factor116 - 1) * bundle_rate116[selected116]
        targets116.append(yy116)
        groups116.extend([(drugname116, factor116)] * len(selected116))
D116 = np.concatenate(contrast116)
Y116 = np.concatenate(targets116)
groups116 = np.asarray(groups116, dtype=object)
fits116 = {}
summary116 = {}
for head116 in range(4):
    xx116 = np.column_stack([features116[:, head116], np.ones(len(valid116))])
    dx116 = np.column_stack([D116[:, head116], np.zeros(len(D116))])
    sf116 = xx116.T @ xx116 / len(xx116)
    sc116 = dx116.T @ dx116 / len(dx116)
    rhs116 = dx116.T @ Y116 / len(dx116)
    ridge116 = np.eye(193) * 1e-6
    ridge116[-1, -1] = 1e-10
    for penalty116 in (0.03, 0.3, 3.0):
        bb116 = solve(sc116 + penalty116*sf116 + ridge116,
                      rhs116, assume_a="pos")
        fp116, cp116 = xx116 @ bb116, dx116 @ bb116
        fits116[(head116, penalty116)] = bb116
        slopes116 = {}
        for di116, name116 in enumerate(("insulin", "norepinephrine")):
            for fac116 in (0.0, 0.5, 2.0, 3.0):
                mm116 = ((groups116[:, 0] == name116) &
                         (groups116[:, 1] == fac116))
                yt116 = Y116[mm116, di116]
                slopes116[(name116, fac116)] = round(
                    float(yt116 @ cp116[mm116, di116] / (yt116 @ yt116)), 3)
        summary116[(head116, penalty116)] = {
            "factual_RMS": np.sqrt(np.mean(fp116**2, axis=0)).round(3).tolist(),
            "contrast_RMSE": np.sqrt(np.mean((cp116-Y116)**2, axis=0)).round(3).tolist(),
            "slopes": slopes116,
        }
        print(head116, penalty116, summary116[(head116, penalty116)])
print("contrast shape:", D116.shape)
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 116 / attention dose-residual extraction\n"
            "Action attention is strong (head 2≈0.78, head 3≈0.66); latent attention "
            "is negligible. Tested analytical dose-contrast extractors on the four "
            "attention heads, penalizing factual signals and including cross-drug "
            "zero targets. Results: " + repr(summary116) + "\nNo weights edited.\n")
```
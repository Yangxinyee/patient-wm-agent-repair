The replay and readout checks matched their caches closely, and the small perturbations produced stable responses. The private coordinates also exactly reconstruct their fitted gated innovations. Next I’ll use forward-only finite differences to fit two token-response directions by regression, then check them on held stays. Weak or wrong-sign held responses would reject the calibration.

```python
import time
start215 = time.time()
W = Wcal214

# Confirm that action inputs align with the private-posterior calibration rows.
Ubase215 = encode_actions168((1.0, 1.0))
Xbase215 = torch.cat([context, e_cal161, Ubase215], dim=1)
print("ACTION INPUT ALIGNMENT RMS",
      float(torch.mean((Xbase215-input196)**2).sqrt()))

post0w215 = torch.as_tensor(np.asarray(load("z_post.0.weight"))).float()
post0b215 = torch.as_tensor(np.asarray(load("z_post.0.bias"))).float()
post2w215 = torch.as_tensor(np.asarray(load("z_post.2.weight"))).float()
post2b215 = torch.as_tensor(np.asarray(load("z_post.2.bias"))).float()

def private_posterior215(inputs):
    return F.linear(
        F.gelu(F.linear(inputs, post0w215, post0b215)),
        post2w215, post2b215
    )[:, :32]

zbase215 = private_posterior215(Xbase215)
print("PRIVATE POSTERIOR RECONSTRUCTION RMS",
      np.sqrt(np.mean(
          (array196(zbase215)-array196(private_z209))**2)))
print("SELECTED DRUG COUNTS",
      dict(zip(*np.unique(np.asarray(drugs182), return_counts=True))))

for drug215, column215 in [("insulin", 0), ("norepinephrine", 1)]:
    select215 = np.asarray(drugs182) == drug215
    rr215 = array196(rates182)[select215]
    for scale215 in [0.0, 0.5, 1.5, 2.0, 3.0]:
        scales215 = [1.0, 1.0]
        scales215[column215] = scale215
        uu215 = encode_actions168(tuple(scales215))
        zz215 = private_posterior215(
            torch.cat([context, e_cal161, uu215], dim=1)
        )
        dz215 = array196(
            zz215[ids182, [16, 12][column215]]
            - zbase215[ids182, [16, 12][column215]]
        )[select215]
        if scale215 != 1.0:
            dr215 = (scale215-1.0)*rr215
            print("PRIVATE DOSE CONTRAST", drug215, scale215,
                  "slope", np.dot(dr215, dz215)/np.dot(dr215, dr215),
                  "RMS error", np.sqrt(np.mean((dz215-dr215)**2)))

# Measure the local response of all four sensor means to the 192 token coordinates.
epsilon215 = 0.025
J215 = np.empty((n182, 4, 192), dtype=np.float64)
identity215 = np.eye(192, dtype=np.float32)
for start_dim215 in range(0, 192, 16):
    stop_dim215 = min(start_dim215+16, 192)
    directions215 = identity215[start_dim215:stop_dim215]
    nd215 = len(directions215)
    indices215 = np.tile(np.arange(n182), 2*nd215)
    shifts215 = np.repeat(
        epsilon215*np.concatenate([directions215, -directions215]),
        n182, axis=0
    )
    hh215, oo215 = replay_token214(
        shifts215, indices215, batch_size=128
    )
    mm215, _ = readout214(
        official_h182[indices215]+hh215-base_h214[indices215],
        official_o182[indices215]+oo215-base_o214[indices215],
        indices215
    )
    responses215 = array196(mm215).reshape(2, nd215, n182, 4)
    J215[:, :, start_dim215:stop_dim215] = (
        (responses215[0]-responses215[1])/(2*epsilon215)
    ).transpose(1, 2, 0)

print("JACOBIAN MEAN-DIRECTION NORMS",
      np.linalg.norm(J215.mean(0), axis=1))
print("JACOBIAN ROW-NORM MEDIANS",
      np.median(np.linalg.norm(J215, axis=2), axis=0))

# Estimate native per-rate responses from the cached dose contrasts.
native215 = {}
directions_fit215 = {}
fit_summaries215 = []
drug_labels215 = np.asarray(drugs182)
rate_vector215 = array196(rates182).reshape(-1)
hold215 = np.asarray(cal_hold183, dtype=bool)
causal_targets215 = {
    "insulin": np.array([-1.988349, -2.05, 0.0, 0.0]),
    "norepinephrine": np.array([0.0, 0.0, 3.288830, 3.10])
}

for drug215 in causal_targets215:
    mask215 = drug_labels215 == drug215
    numerator215 = np.zeros((n182, 4))
    denominator215 = np.zeros(n182)
    for cc215 in contrasts182:
        if cc215["drug"] == drug215:
            dr215 = (float(cc215["scale"])-1.0)*rate_vector215
            numerator215[mask215] += (
                dr215[mask215, None]
                * array196(cc215["original_shift"])[mask215]
            )
            denominator215[mask215] += dr215[mask215]**2
    slope215 = numerator215 / np.maximum(
        denominator215[:, None], 1e-12
    )
    native215[drug215] = slope215
    train215 = mask215 & ~hold215
    test215 = mask215 & hold215
    A215 = J215[train215].reshape(-1, 192)
    y215 = (
        causal_targets215[drug215][None, :]
        - slope215[train215]
    ).reshape(-1)
    covariance215 = A215.T@A215/len(A215)
    cross215 = A215.T@y215/len(A215)
    ridge_unit215 = np.trace(covariance215)/192
    print("NATIVE UNIT SLOPES", drug215,
          np.mean(slope215[mask215], axis=0),
          "fit/held rows", train215.sum(), test215.sum())
    for penalty215 in [0.01, 0.1, 1.0, 10.0]:
        v215 = np.linalg.solve(
            covariance215
            + penalty215*ridge_unit215*np.eye(192),
            cross215
        )
        pred215 = slope215 + np.einsum("ned,d->ne", J215, v215)
        directions_fit215[(drug215, penalty215)] = v215
        summary215 = {
            "drug": drug215, "penalty": penalty215,
            "norm": float(np.linalg.norm(v215)),
            "train_mean": pred215[train215].mean(0),
            "held_mean": pred215[test215].mean(0),
            "held_rmse": np.sqrt(np.mean(
                (pred215[test215]-causal_targets215[drug215])**2, axis=0
            ))
        }
        fit_summaries215.append(summary215)
        print(summary215)

np.savez(
    "/work/token_directions215.npz",
    J=J215, hold=hold215, drugs=drug_labels215,
    **{
        f"{drug215}_{penalty215:g}": vector215
        for (drug215, penalty215), vector215
        in directions_fit215.items()
    }
)
print("SECONDS", round(time.time()-start215, 2))
with open("/work/notes.md", "a") as f215:
    f215.write(
        "\n## Python 215 — forward-only token-response regression\n"
        "- Measured sensor-mean finite differences for every latent-token coordinate on 258 anchored action contexts.\n"
        "- Fitted ridge response directions against record-derived proximal effects, subtracting native dose response.\n"
        "- Stay-held checks and explicit x0–x3 innovation scaling are diagnostics, not rollout certification.\n"
        "- Saved token_directions215.npz; installed neutral weights unchanged.\n"
    )
```
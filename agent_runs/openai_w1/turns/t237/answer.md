The relay has two unused channels. However, the residual cache just inspected predates the stronger presence gates, so its inactive-drug leakage is not a reliable measure of the saved carrier. I’ll verify the installed carrier and test it with the relay before adding more machinery. Material rollout changes would support the transmission hypothesis.

```python
import torch.nn.functional as F

candidate211_baseline = {
    n: load(n).detach().cpu().clone() for n in edited()
}
install_carrier200(12.0, 6.0)

def posterior211(x):
    x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    h = F.gelu(F.linear(x, load("z_post.0.weight"), load("z_post.0.bias")))
    return F.linear(h, load("z_post.2.weight"), load("z_post.2.bias"))[:, :32].numpy()

z211 = posterior211(rp_input)
dims211 = np.asarray(lc_dims, dtype=int).tolist()
print("Carrier coordinates:", dims211)
for j, drug in enumerate(("insulin", "norepinephrine")):
    m = np.asarray(rp_cf_masks[drug], dtype=bool)
    q = z211[:, dims211[j]]
    print(drug,
          "factual active quantiles", np.quantile(q[m], [0, .05, .5, .95, 1]),
          "inactive RMS", float(np.sqrt(np.mean(q[~m]**2))))
    for mult in (0., .5, 1.5, 2., 3.):
        qcf = posterior211(rp_cf_input[(drug, mult)])[:, dims211[j]]
        print(" ", mult, "active contrast quantiles",
              np.quantile((qcf-q)[m], [0, .1, .5, .9, 1]))

print("Edited tensors:", edited())
print("Forward stays:", len(relay207_stays))
candidate211_weights = {n: load(n).detach().cpu().clone() for n in edited()}

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 211\n"
            "Installed saved strong-gate latent carrier (insulin gain 12, NE gain 6) "
            "on top of compensated uniform attention relay. Verifying carrier residuals "
            "from installed weights rather than older ungated caches. This is a reversible "
            "transmission probe, not yet a validated repair. "
            "Counts: 211 Python runs, 19 forwards.\n")
```
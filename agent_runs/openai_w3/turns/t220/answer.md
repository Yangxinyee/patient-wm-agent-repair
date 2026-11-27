The pilot’s baseline glucose rollout drifts upward by about 4.6 units. Its local insulin response is fairly symmetric, but norepinephrine transport saturates and affects both glucose sensors. I’ll trace scaled doses through the actual prelude and inspect the isolated carriers. This will test whether normalization or compensation weakens the signal before recurrent feedback begins.

```python
print("=== CARRIER ISOLATION ===")
print("reserved posterior coordinates:", reserved154)
print("reserved hidden neurons:", rows154)
print("carrier Gram:", np.round(carrier154.T @ carrier154, 5))
print("carrier projection of z_in:",
      np.round(carrier154.T @ zin154, 5))
print("carrier projection of bias:",
      np.round(carrier154.T @ zb154, 7))
print("extractor projection of z_in:",
      np.round(value_extract154.T @ zin154, 5))
print("Y154 first rows:", np.round(Y154[:8], 4))

print("\n=== ACTUAL PRELUDE DOSE CONTRASTS ===")
p0_198 = prelude155(tokens192, checkpoint193)
after0_198 = p0_198[1]
raw0_198 = after0_198 + torch.nn.functional.linear(
    hidden192(after0_198),
    tensor153(checkpoint193["core.prelude.0.ffn.2.weight"]),
    tensor153(checkpoint193["core.prelude.0.ffn.2.bias"]))
dual198 = np.linalg.pinv(directions187).T

scaled_u198 = {}
scaled_tokens198 = {}
for d in [0, 1]:
    for s in [0.0, 0.5, 2.0, 3.0]:
        factors = [1.0, 1.0]
        factors[d] = s
        uu = action_vectors87(tuple(factors))
        zz = posterior153(C154, E154, uu, post187)
        tt = tokens192.clone()
        tt[:, 5] = tensor153(uu)
        tt[:, 6] = torch.nn.functional.linear(
            zz, tensor153(zin154), tensor153(zb154))
        scaled_u198[(d, s)] = uu
        scaled_tokens198[(d, s)] = tt
        pp = prelude155(tt, checkpoint193)
        aa = pp[1]
        rr = aa + torch.nn.functional.linear(
            hidden192(aa),
            tensor153(checkpoint193["core.prelude.0.ffn.2.weight"]),
            tensor153(checkpoint193["core.prelude.0.ffn.2.bias"]))
        dy = np.asarray(pairs187[(d, s)][1])
        use = np.asarray(valid187, bool) & (np.abs(dy[:, d]) > 1e-8)
        attcoef = arr183(aa-after0_198).mean(axis=1) @ dual198
        rawcoef = arr183(rr-raw0_198).mean(axis=1) @ dual198
        target = dy[use, d]
        print(d, s, "target mean", round(float(target.mean()), 5),
              "attention/raw coefficients",
              np.round([attcoef[use, d].mean(), rawcoef[use, d].mean()], 5),
              "raw/target slope",
              round(float(rawcoef[use, d] @ target / (target @ target)), 4),
              "E change RMS",
              round(float(np.sqrt(np.mean(arr183(pp[0]-p0_198[0])[use]**2))), 5))

print("\n=== RECURRENT HELPER SIGNATURE ===")
print(inspect.signature(state_update82), state_update82.__code__.co_names)
print("response186 variables:", response186.__code__.co_varnames)

print("\n=== ORIGINAL VS CURRENT COMPENSATION NORMS ===")
for name in ["core.prelude.0.ffn.2.weight", "core.prelude.0.ffn.2.bias",
             "z_in.weight", "z_post.2.weight"]:
    print(name, float(np.linalg.norm(
        np.asarray(checkpoint193[name]) - as_numpy(load_original(name)))))

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 198; forwards 12. Baseline pilot glucose rollout drifts +4.55 units. Local insulin transport is symmetric, while NE transport saturates and crosses into glucose. Tracing scaled action signals through actual attention and FFN, and inspecting exact carrier isolation. No weights changed.\n")
```
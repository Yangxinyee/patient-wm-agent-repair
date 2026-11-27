The retained caches contain paired dose edits, so we can isolate dose changes while holding history fixed. I’ll rebuild those inputs with the original posterior and calibrate small state directions against the record-derived effects. This approach is worth testing if the directions give the right signs without strong cross-sensor leakage.

```python
post_orig278 = {n: load_original(n) for n in [
    "z_post.0.weight", "z_post.0.bias", "z_post.2.weight", "z_post.2.bias"
]}
def original_tokens278(u):
    u = tensor153(u)
    z = posterior153(C154, E154, u, post_orig278)
    x = tensor153(tokens259).clone()
    x[:, 5, :] = u
    x[:, 6, :] = F.linear(z, load_original("z_in.weight"), load_original("z_in.bias"))
    return x

base_tokens278 = original_tokens278(u_factual255)
base_parts278 = parts254(base_tokens278, original=True)
print("Original prelude return shapes:", [tuple(x.shape) for x in base_parts278])
print("Return instructions:", [
    (i.opname, i.argrepr) for i in list(dis.get_instructions(parts254))[-10:]
])
assert tuple(base_parts278[1].shape) == (4213, 4, 768)
raw_check278 = base_parts278[0] + F.linear(
    base_parts278[1], load_original("core.prelude.0.ffn.2.weight"),
    load_original("core.prelude.0.ffn.2.bias"))
print("Raw-output reconstruction error:",
      float(torch.max(torch.abs(raw_check278-base_parts278[2]))))
assert float(torch.max(torch.abs(raw_check278-base_parts278[2]))) < 1e-4

dose_parts278 = {}
for key, tokens in fresh_tokens262.items():
    changed_tokens = original_tokens278(tensor153(tokens)[:, 5, :])
    dose_parts278[key] = parts254(changed_tokens, original=True)
print("Dose contrast keys:", list(dose_parts278))
print("Recorded dose ranges:", np.min(rates267, axis=0), np.max(rates267, axis=0))
print("Record-derived sensor targets:", np.asarray(target255))

a278 = arr183(base_tokens278[:, :4, :]).reshape(4213, -1).astype(np.float64)
b278 = arr183(tokens_orig169[:, :4, :]).reshape(16, -1).astype(np.float64)
dist278 = np.maximum(
    np.sum(b278*b278, axis=1)[:, None] + np.sum(a278*a278, axis=1)[None, :]
    - 2*b278@a278.T, 0)
indices278 = np.argmin(dist278, axis=1)
print("Calibration-state matching errors:", dist278[np.arange(16), indices278])

saved_r233_278 = r233_raw
r233_raw = base_parts278[2][indices278].clone()
eps278 = 0.025
plus278 = np.asarray(response_raw234(np.eye(192, dtype=np.float32)*eps278))
minus278 = np.asarray(response_raw234(-np.eye(192, dtype=np.float32)*eps278))
Jorig278 = ((plus278-minus278)/(2*eps278)).T
print("Original raw-state response Jacobian shape:", Jorig278.shape)

sensor_weights278 = np.array([1., .5, 1., .5])
Jweighted278 = Jorig278 * sensor_weights278[:, None]
targetweighted278 = np.asarray(target255) * sensor_weights278[:, None]
ridge278 = 1e-4 * np.trace(Jweighted278 @ Jweighted278.T) / 4
directions278 = Jweighted278.T @ np.linalg.solve(
    Jweighted278 @ Jweighted278.T + ridge278*np.eye(4), targetweighted278)
print("Direction norms:", np.linalg.norm(directions278, axis=0))
print("Linear sensor effects:", Jorig278 @ directions278)
test_injections278 = np.vstack([
    directions278[:, 0], -directions278[:, 0],
    .1*directions278[:, 1], -.1*directions278[:, 1]
]).astype(np.float32)
print("Finite sensor responses (+/-1 insulin, +/-0.1 NE):",
      np.asarray(response_raw234(test_injections278)))
r233_raw = saved_r233_278

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 278; forwards 24. Rebuilt exact original-posterior factual and paired changed-dose tokens, retaining original prelude features. Calibrated raw-state directions against record-only four-sensor effects using central finite differences; no gradients or weight changes. Preparing simpler original-prelude FFN correction.\n")
```
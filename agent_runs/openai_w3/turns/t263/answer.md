I found a sensor-ordering error in the diagnostic target matrix: it assigned insulin to MAP cuff and norepinephrine to glucose lab. Those full-sensor trials are invalid; the record estimates are unaffected. I will correct the ordering and isolate the installed dose directions before making a reversible replacement.

```python
r235_targets = np.array([
    [0.0, 3.2],       # map_cuff
    [-1.94, 0.0],     # glucose_finger
    [0.0, 3.2],       # map_arterial
    [-1.94, 0.0],     # glucose_lab
], dtype=np.float64)
print("installed primary linear targets", np.round(J186 @ directions186, 4))
print("correct sensor-ordered targets", r235_targets)

r235_directions = {}
for ridge in [1e-6, 1e-5, 3e-5]:
    d = r234_J.T @ np.linalg.solve(
        r234_J @ r234_J.T + ridge*np.eye(4), r235_targets)
    r235_directions[ridge] = d
    tests = np.stack([-d[:, 0], d[:, 0], -0.5*d[:, 1], 0.5*d[:, 1]])
    print("ridge", ridge, "norms", np.round(np.linalg.norm(d, axis=0), 2))
    print("responses (-I,+I,-NE/2,+NE/2)",
          np.round(response_raw234(tests)-r234_baseline, 3))

r235_after = arr183(prelude155(tokens206, overrides=base_override228)[1])
r235_normfact = arr183(F.layer_norm(
    tensor153(tokens206[:, -1]), (192,),
    load_original("core.prelude.0.norm1.weight"),
    load_original("core.prelude.0.norm1.bias")))
r235_Xparts, r235_Yparts = [], []
for key in [(0, 0.5), (0, 2.0), (1, 0.5), (1, 2.0)]:
    tok = r214_tokens[key]
    aft = arr183(prelude155(tok, overrides=base_override228)[1])
    normz = arr183(F.layer_norm(
        tensor153(tok[:, -1]), (192,),
        load_original("core.prelude.0.norm1.weight"),
        load_original("core.prelude.0.norm1.bias")))
    ds = (normz-r235_normfact) @ restricted199
    r235_Xparts.append(np.repeat(ds, 4, axis=0))
    r235_Yparts.append((aft-r235_after).reshape(-1, 192))
r235_X = np.concatenate(r235_Xparts)
r235_Y = np.concatenate(r235_Yparts)
r235_transport = np.linalg.lstsq(r235_X, r235_Y, rcond=None)[0].T
r235_err = r235_Y-r235_X @ r235_transport.T
print("installed transport norms", np.linalg.norm(r235_transport, axis=0))
print("paired transport residual/total RMS",
      np.sqrt(np.mean(r235_err**2)), np.sqrt(np.mean(r235_Y**2)))
print("transport vs old primary relative error",
      np.linalg.norm(r235_transport-directions186)/np.linalg.norm(directions186))

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 235; forwards 16. Found sensor-order bug in T4_228: correct order is cuff, finger, arterial, lab. Full-sensor diagnostic trials were mis-targeted; current primary repair and record estimates unchanged. Computed correctly ordered raw directions and isolated actual paired-dose attention transport. No weights changed.\n")
```
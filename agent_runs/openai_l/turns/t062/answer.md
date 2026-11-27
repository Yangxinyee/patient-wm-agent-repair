The measurement encoder uses float64 bin boundaries, whereas float32 boundaries were essential to matching action bundles. I’ll test that discrepancy directly and report posterior error by bundle type. A large drop on measurement bundles, with action bundles remaining matched, would confirm that binning—not the action equations—is the remaining problem.

```python
import torch
import torch.nn.functional as F

def observation_embeddings56(gu=1, mu=2, me=1, ie=4, he=3,
                             boundary32=True, side="left"):
    n = len(obs_codes51)
    nums = np.zeros((n, 6), dtype=np.float32)
    bins = {a: np.zeros(n, dtype=np.int64) for a in ("value", "dose", "rate")}
    for code in ("glucose_finger", "glucose_lab", "map_arterial", "map_cuff"):
        mask = (obs_codes51 == code) & np.isfinite(obs_values51)
        spec = num42[(code, "value")]
        v = obs_values51[mask]
        nums[mask, 0] = np.clip(
            (v.astype(np.float64) - spec["median"]) / spec["iqr"], -10, 10)
        nums[mask, 1] = 1
        dtype = np.float32 if boundary32 else np.float64
        bins["value"][mask] = 1 + np.searchsorted(
            np.asarray(spec["fine_bin_edges"], dtype=dtype),
            np.asarray(v, dtype=dtype), side=side)
    units = np.zeros(n, dtype=np.int64)
    units[np.isin(obs_codes51, ["glucose_finger", "glucose_lab"])] = gu
    units[np.isin(obs_codes51, ["map_arterial", "map_cuff"])] = mu
    ek = np.full(n, me, dtype=np.int64)
    ek[obs_codes51 == "hospital_admission"] = he
    ek[obs_codes51 == "icu_admission"] = ie
    ek[np.isin(obs_codes51, ["insulin", "norepinephrine"])] = 2
    emb = (
        wt42("embed.code.weight")[torch.tensor([code42[c] for c in obs_codes51])]
        + wt42("embed.kind.weight")[torch.tensor(obs_kinds51)]
        + wt42("embed.ekind.weight")[torch.tensor(ek)]
        + defaults42
        + wt42("embed.categorical.unit.weight")[torch.tensor(units)]
        + wt42("embed.categorical.rate_unit.weight")[0]
        + wt42("embed.kq.weight")[0]
    )
    for attr in ("value", "dose", "rate"):
        emb = emb + wt42(f"embed.bins.{attr}.weight")[torch.tensor(bins[attr])]
    return emb + F.linear(torch.tensor(nums), wt42("embed.numeric.weight"),
                          wt42("embed.numeric.bias"))

def evaluate56(**kw):
    e = encode_set50(observation_embeddings56(**kw), obs_owner51,
                     obs_counts51, "encode")
    hidden = F.gelu(affine42(torch.cat(
        [ctx51, e[valid_indices51], u_all51[valid_indices51]], dim=1),
        "z_post.0"))
    zp = affine42(hidden, "z_post.2")[:, :32]
    return float(torch.mean((zp - target_z51)**2).sqrt()), e, hidden, zp

results56 = []
for b32 in (False, True):
    for side56 in ("left", "right"):
        for me56 in range(wt42("embed.ekind.weight").shape[0]):
            rr56 = evaluate56(boundary32=b32, side=side56, me=me56)
            results56.append((rr56[0], b32, side56, me56))
print("Best boundary/event-kind configurations:", sorted(results56)[:8])
best56 = min(results56)
fit56 = evaluate56(boundary32=best56[1], side=best56[2], me=best56[3])
err56 = ((fit56[3] - target_z51)**2).mean(dim=1).numpy()
group56 = np.full(len(s51["hour"]), "other", dtype=object)
for lab56, mask56 in [
    ("measurement", np.isin(obs_codes51, ["glucose_finger", "glucose_lab",
                                        "map_arterial", "map_cuff"])),
    ("visit", np.isin(obs_codes51, ["hospital_admission", "icu_admission"])),
    ("END", obs_kinds51 == 2),
]:
    group56[np.unique(obs_owner51[mask56])] = lab56
group56[(action_counts51 > 0) & (group56 == "other")] = "action_only"
group56[(action_counts51 > 0) & (group56 == "END")] = "action_and_END"
for lab56 in np.unique(group56[valid_indices51]):
    mask56 = group56[valid_indices51] == lab56
    print(lab56, "n=", int(mask56.sum()),
          "posterior_RMS=", float(np.sqrt(err56[mask56].mean())),
          "max_bundle_RMS=", float(np.sqrt(err56[mask56].max())))
print("Example observation metadata:", obs_meta51[:5])
print("Largest-error bundles:",
      [(int(s51["stay_id"][valid_indices51[i]]),
        float(s51["hour"][valid_indices51[i]]),
        str(group56[valid_indices51[i]]), float(np.sqrt(err56[i])))
       for i in np.argsort(err56)[-8:]])
with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 53–56 / recovery and measurement bin audit\n"
            "Recovered architecture and causal evidence. No tensors edited. "
            "Testing float32 measurement boundaries, independently by bundle type. "
            f"Best posterior test: {best56}; errors retained in err56.\n")
```

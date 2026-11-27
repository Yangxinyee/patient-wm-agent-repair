Starts reconstruct closely, but the assumed ending encoding fails badly. No edit is justified yet. I’ll test whether endings enter the observation encoder or are excluded from the posterior, and correct the rounded timestamps. Numerical agreement across all bundle types is the criterion for proceeding.

```python
def latent_context(indices):
    indices = np.asarray(indices, dtype=np.int64)
    prev = torch.from_numpy(latent_state[indices-1, :960])
    seconds = np.round(latent_hour * 3600)
    dt = torch.from_numpy((seconds[indices] - seconds[indices-1]).astype(np.float32))
    hf = prev[:, 192:].reshape(-1, 4, 192) * torch.exp(
        -dt[:, None, None] * torch.exp(repair_weights["core.flow.a"])[None, None, :])
    return (prev[:, :192] + repair_norm(hf.mean(1), "core.out_norm")
            + repair_mlp_act(torch.log(dt[:, None]) / 10, gap_prefix41, F.gelu))

pred42 = repair_linear(latent_hidden(latent_context(idx41), u41), "z_post.2")[:, :32]
print("Rounded-time start RMS:",
      float(torch.mean((pred42-torch.from_numpy(latent_state[idx41, 960:]))**2).sqrt()))

end_idx42 = np.array([i for i, _ in single_end41[:40]])
end_sources42 = [r for _, r in single_end41[:40]]
end_ctx42 = latent_context(end_idx42)
end_target42 = torch.from_numpy(latent_state[end_idx42, 960:])
print("END flags:", {
    "insulin": int(state_archive40["has_insulin"][end_idx42].sum()),
    "norepinephrine": int(state_archive40["has_norepinephrine"][end_idx42].sum())
})
print("END target standard deviation:", float(end_target42.std()))

def latent_encode_batch(atoms, encoder):
    pooled = repair_mlp_act(atoms, encoder + ".mlp1", F.gelu)
    pooled = pooled + repair_linear(
        torch.full((atoms.shape[0], 1), np.log(2), dtype=torch.float32),
        encoder + ".size")
    return repair_norm(repair_mlp_act(pooled, encoder + ".mlp2", F.gelu),
                       encoder + ".norm")

def latent_post_mean(context, e, u):
    hidden = F.gelu(repair_linear(torch.cat([context, e, u], -1), "z_post.0"))
    return repair_linear(hidden, "z_post.2")[:, :32]

end_search42 = []
for empty42 in [latent_empty_e.expand(len(end_idx42), -1), torch.zeros_like(end_ctx42)]:
    zz42 = latent_post_mean(end_ctx42, empty42, torch.zeros_like(end_ctx42))
    print("END ignored, e norm", float(empty42.norm(dim=-1).mean()),
          "RMS", float(((zz42-end_target42)**2).mean().sqrt()))

for branch42, numbers42, unit_mode42, rate_mode42, action42, status42, ekind42 in itertools.product(
        ["observation", "action"], ["absent", "copied"],
        ["absent", "canonical"], ["absent", "canonical"], [0, 1], [0, 1], [0, 2]):
    atoms42 = []
    for source42 in end_sources42:
        insulin42 = source42["event"] == "insulin"
        av42 = repair_atom_base(source42, kind=2, include_numbers=numbers42 == "copied")
        if ekind42 != 2:
            av42 = av42 - repair_weights["embed.ekind.weight"][2] + repair_weights["embed.ekind.weight"][ekind42]
        av42 = av42 + latent_categories(
            (3 if insulin42 else 4) if unit_mode42 == "canonical" else 0,
            (2 if insulin42 else 3) if rate_mode42 == "canonical" else 0,
            action=action42, status=status42)
        atoms42.append(av42)
    aa42 = torch.stack(atoms42)
    if branch42 == "observation":
        ee42 = latent_encode_batch(aa42, "encode")
        uu42 = torch.zeros_like(ee42)
    else:
        ee42 = latent_empty_e.expand(len(aa42), -1)
        uu42 = latent_encode_batch(aa42, "encode_action")
    zz42 = latent_post_mean(end_ctx42, ee42, uu42)
    rms42 = float(((zz42-end_target42)**2).mean().sqrt())
    end_search42.append((rms42, branch42, numbers42, unit_mode42,
                         rate_mode42, action42, status42, ekind42))
print("Best END conventions:", sorted(end_search42)[:10])

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 42; forward 6 — ending-path correction\n"
            "- Initial END-as-action reconstruction failed; no weights changed.\n"
            f"- Expanded END search: {sorted(end_search42)[:3]}.\n")
```
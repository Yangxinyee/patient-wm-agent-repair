The architecture confirms that the signal passes through several nonlinear stages, so the local checks cannot substitute for rollout testing. The candidate is retained despite the truncated inventory. I’ll verify and install it reversibly, then test the same stays as the earlier pilot; improved dose symmetry must come without exceeding the factual-loss limit.

```python
import pandas as pd

r209_snapshot = {
    n: torch.as_tensor(load(n), dtype=torch.float32).clone()
    for n in edited()
}
r209_candidate = {
    n: torch.as_tensor(v, dtype=torch.float32).clone()
    for n, v in candidate206.items()
}

print("=== CANDIDATE CHECK ===")
for n, t in r209_candidate.items():
    assert tuple(t.shape) == tuple(SHAPES[n]), (n, t.shape, SHAPES[n])
    assert torch.isfinite(t).all(), n
    cur = torch.as_tensor(load(n), dtype=torch.float32)
    cp = torch.as_tensor(checkpoint193[n], dtype=torch.float32)
    print(n, "change/current", float(torch.linalg.vector_norm(t-cur)),
          "change/checkpoint193", float(torch.linalg.vector_norm(t-cp)))

for n, t in r209_candidate.items():
    save_edit(n, t)
installed206 = True

print("\n=== TEST STAYS ===")
print("fact193 columns:", list(fact193.columns))
r209_stay_col = next(c for c in fact193.columns if "stay" in c.lower())
r209_stays = [int(x) for x in fact193[r209_stay_col].drop_duplicates().iloc[:100]]
print("stays:", r209_stays)
print("installed tensors:", edited())

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n\nPython run 209; forwards 12. Verified candidate206 tensor shapes "
        "and finiteness, retained reversible r209_snapshot, and installed "
        "the restricted-carrier/action-only gate candidate. Next tests use "
        "the same 100 stays as pilot193. Acceptance requires factual NLL "
        "increase <=0.005 and improved common-random-number dose effects.\n"
    )
```
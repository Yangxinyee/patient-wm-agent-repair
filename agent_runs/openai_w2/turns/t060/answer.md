The earlier decoder experiment found that dose information was too weak at the final head to amplify safely. I’ll now test the latent posterior’s action input, downstream of the normalized encoder: revert numeric scale 12 and multiply only those weights by four. A stronger, correctly signed insulin response would support this route; little change or a sign reversal would refute it.

```python
numeric12_backup = {n: load(n).clone() for n in edited()}
for n in list(edited()):
    drop_edit(n)

posterior_action_scale = 4.0
posterior_w0 = load_original("z_post.0.weight").clone()
posterior_candidate = posterior_w0.clone()
posterior_candidate[:, 384:576] *= posterior_action_scale
save_edit("z_post.0.weight", posterior_candidate)

print("Edited tensors:", edited())
print("Original posterior block norms:",
      [float(posterior_w0[:, j:j+192].norm()) for j in (0, 192, 384)])
print("Candidate action-block norm:",
      float(posterior_candidate[:, 384:576].norm()))
print("Benchmark stay count:", len(benchmark_stays))
with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 50: posterior-action amplification diagnostic\n"
            "Reverted scale-12 embed.code/numeric edits; backed them up as numeric12_backup. "
            "Only z_post.0.weight edited: action-input columns 384:576 multiplied by 4. "
            "This tests whether amplification downstream of action normalization "
            "can recover dose sensitivity. No conclusion until exact rollout and factual tests.\n")
```

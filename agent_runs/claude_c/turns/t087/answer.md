The 20x run settles the question: the action encoder saturates. Insulin x2 came out at −2.11, slightly weaker than the −2.54 seen at 5x. Insulin x1/2 was +1.50 against +1.69 at 5x, and both norepinephrine readouts sat near zero. Scaling the dose inputs further is a dead end.

The reason is structural. A dose-specific edit can only remap a layer's inputs or outputs in a space whose directions I can actually observe. The 32-d z posterior is such a space.

Any linear layer that reads z can be remapped, W ← W·M, as a short sequence of rank-1 operations; I already did this once for `z_in`. Before, I mapped the dose signal onto z26 itself, and the state ignored it. A more promising mapping sends the drug-bundle dose signal p onto the z-direction g that encodes "this reading came out higher or lower than predicted". If the state update relies on that direction, the dose would move the state the way a glucose surprise does, which should carry far more weight. In teacher-forced mode the next real reading would then overwrite most of that shift, which limits the factual cost.

Before spending a forward pass, I'll check from saved files whether such a g exists. If the z posterior at glucose-reading bundles is strongly explained by the reading's surprise, the lever is plausible; if not, I drop it.

```python
with open('/work/notes.md','a') as f:
    f.write("\n## Restart-4 (runs 74-78, pass 9)\n- Pass 9: numeric z_dose/z_rate cols x20: ins x2 -2.11, half +1.50, ne x2 +0.15, half +0.00 -> input-side scaling SATURATES (x5 was -2.54). Dead end.\n- Idea: remap z-space in z_in (W<-W M, M=I+k g p^T): dose signal p (drug bundles) -> glucose-surprise direction g (reading bundles). Checking if g exists.\n")
z3 = np.load('/work/fwd_3/states_original.npz', allow_pickle=True)
print({k_: z3[k_].shape for k_ in z3.files})
fa2 = pd.read_csv('/work/fwd_2/factual_original.csv')
print(fa2.columns.tolist()); print(fa2.head(3).to_string())
```
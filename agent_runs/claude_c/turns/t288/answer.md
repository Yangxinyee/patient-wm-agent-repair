Run 225 shows how each drug atom is built: its code row, kind rows, unit and rate-unit rows, the dose and rate bin rows, and the shared numeric term. The drug atoms are pooled by a DeepSets encoder (MLPs) and then a LayerNorm. Only the code row and the unit and rate-unit rows differ between insulin and norepinephrine. The numeric and bin terms are shared, so N5 necessarily amplifies both drugs together.

My working conclusion is that the input side is capped near N5. The only drug-specific handles left are those constant rows. Shrinking norepinephrine's constant part could make its dose term relatively stronger, which would raise its dose sensitivity alone.

Before spending passes on that, I need two facts:
- whether the code rows are also used by the heads in a way that touches the reading NLL or the rollout;
- which categorical rows belong to each drug.

So I'll print ARCHITECTURE.md lines 57-91 (heads and rollout) and the VOCAB code list. If the rollout samples only reading values, and code rows enter value predictions only for the reading's own code, then a norepinephrine-row edit only touches its drug token and is worth testing. Otherwise I drop the idea.

```python
print('\n'.join(f"{i}: {A[i][:210]}" for i in range(57,92) if A[i].strip())[:3200])
print(VOCAB['codes'] if isinstance(VOCAB.get('codes'),list) else list(VOCAB['codes'])[:15])
```
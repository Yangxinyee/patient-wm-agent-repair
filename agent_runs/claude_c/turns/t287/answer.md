Run 224 confirmed the update path. The drug token u_k enters only the prelude attention, alongside the slot tokens h+slots, the bundle token e_k and the z token. The prelude output is renormalised by prelude_norm before it touches the state. Since |h| is about 840, the drug token's injection is diluted, which matches the weak dose response I measured.

Before deciding whether anything beyond N5 is worth a forward pass, I want to know how the drug token is built. In particular: do drug-specific constant embeddings (code row, unit, rate_unit) enter it, and does e_k also see drug events? If the drug token mixes a large drug-specific constant with the dose part, shrinking that constant for one drug could raise its dose sensitivity separately from the other drug.

I'll print the first 33 lines of ARCHITECTURE.md, truncated.

```python
print('\n'.join(f"{i}: {A[i][:200]}" for i in range(0,33) if A[i].strip())[:3000])
```
This changes the picture. The prelude is a pre-norm transformer layer over [h + slots, e, u, z token]. Its slot outputs are layer-normalised into E. Because h is very large (norm about 1200), anything the prelude adds from the e, u or z tokens is diluted to almost nothing: z+2 moved h by 0.003. That explains the original model's weak dose response, since doses enter only through the u token in this prelude. It also means the z_in injection route cannot work without large internal edits.

Readings carry strong information because past e tokens enter through the memory that the blocks attend to.

I need a stronger route for dose information. The head adds per-code "last" features (last value, rate-bin and status embeddings) next to the view. If those carry information about the most recent infusion, a head-side edit could read the dose directly. I'll pull the ARCHITECTURE.md lines on the head inputs.

```python
hits = [i for i,l in enumerate(lines) if any(s in l for s in ('last_','bins.rate','status','attr_heads','value_bin','transition','memory','blocks','b_in','delta'))]
for i in hits[:40]: print(i, lines[i][:230])
```
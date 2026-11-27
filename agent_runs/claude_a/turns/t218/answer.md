The grid test explains the bad fits: the encoding of the insulin bundle is identical at every rate from 0.1 to 12, which is impossible if the rate were actually used.

So `encode_bundles` is not reading the rate and dose I edit. Most likely it, or `atom_vectors`, looks up cached per-atom vectors by DataFrame index, and every copy of an atom keeps its index. That would also mean my four "dose-edited" datasets in run 209 were all secretly factual. The identical off-target figures (p99 141.7) and gate minima across all four datasets fit this.

The factual-only R² of 0.25 also clashes with my earlier note (0.995). I will recheck it once the encoding is right.

Next I will inspect the bytecode names and constants of `atom_vectors` and `encode_bundles`, to find which columns or caches they read. If they read precomputed z/bin columns or a cache, I will feed edited atoms through the right inputs, or write an explicit atom-vector builder from VOCAB, which describes the z-score and bin rules.

```python
for f in ['atom_vectors','encode_bundles','encode_set','deepsets','action_atoms','meas_atoms','atomvec1','fbin','fbin_v','zsc','zsc_v']:
    try:
        c_=globals()[f].__code__
        print(f, '| names', c_.co_names[:30], '| consts', [x for x in c_.co_consts if isinstance(x,(str,int,float))][:25])
    except Exception as ex: print(f,'ERR',ex)
```
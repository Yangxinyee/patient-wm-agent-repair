The insulin dose score is z = rate − 1.5 (IQR 1), with bin edges topping out at 5 U/h. A doubled 3 U/h dose lands at z = 4.5, which N5 turns into an input of 22.5, far beyond anything seen in training. The x1/2 edit moves the same stays down to z ≈ 0, which explains why half improves under N5 while x2 collapses at high doses.

The original's per-unit response doesn't fit a single curve of the numeric code. So the bins, which N5 leaves unchanged, must carry a good share of the dose response. To separate the two contributions, I'll print the banded insulin tables for the bins experiments: pass 7 (bin variation x3) and pass 27 (bin 16 set equal to bin 12). If amplified bins mainly change the high-dose bands, a mix of moderate numeric scaling and bin edits could avoid the saturation.

```python
for k,m in [(7,'edited'),(27,'edited'),(18,'edited'),(4,'edited')]:
    f=f'/work/fwd_{k}/dose_edit_{m}.csv'
    if os.path.exists(f): print(k, band_tab(f,'insulin_x2')[0], band_tab(f,'insulin_half')[0])
```
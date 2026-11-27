Splitting by band shows the edits help different doses. N5 boosts low doses (band 1 x2: -0.82 → -2.19) but saturates the high ones. Head0 x4 (pass 18) does the opposite: it boosts mid/high doses (band 3 x2 -3.88, half +6.74) and barely touches low doses. Amplifying the bin variation (pass 7) saturates the high doses too, and flips low-dose x1/2 to the wrong sign.

Every input-side code moves the action token through one bounded total swing, about 7 mg/dL accumulated. So the main readout via inputs alone seems capped near -2.5 to -3. Only downstream potency gains go further, and they cost factual fit. To see whether another head gives a cheaper high-dose boost than head0, I'll print the banded tables for the head1-3 x4 passes (19-21).

```python
for k in [19,20,21,25]:
    f=f'/work/fwd_{k}/dose_edit_edited.csv'
    print(k, band_tab(f,'insulin_x2')[0], band_tab(f,'insulin_half')[0], band_tab(f,'insulin_x2')[1])
```
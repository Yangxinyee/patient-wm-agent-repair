The rate-band split confirms the out-of-range story for NE x2, and narrows it:

- **Cap stays** (rate 1.0, 60% of rows): x2 gives -0.70 against roughly +6.9 expected.
- **0.5-1 band:** x2 gives +0.28 against +4.6.
- **≤0.5 band:** even in range the model is slightly wrong-signed for both edits.
- **NE half:** right sign only at the cap, about 1/3 of the size.

A row-16 remap mainly targets the cap stays, but row 16 is shared with insulin (above 5 U/h, and insulin x2 for rates above 2.5). Before designing it, I need to know exactly which 6 numeric features enter the atom embedding. Some of them might separate insulin from NE, which would let an edit treat the two drugs differently.

```python
for k in ['numeric','atom','encode.']:
    for l in A.split('\n'):
        if k in l: print(l[:170])
```
The bundle count matches (54). Codes are probably integers. Next I'll map code names from `VOCAB`, then print one insulin bundle and one END-atom example compactly, so I can see what fields END atoms carry.

```python
codes=VOCAB['codes']; print(codes if len(str(codes))<400 else str(codes)[:400])
```
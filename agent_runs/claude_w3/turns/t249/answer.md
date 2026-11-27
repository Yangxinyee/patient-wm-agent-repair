That pass ran on only 3 stays: I passed `sel[:3]` by mistake. On those 160 readings, freeing dimensions 49 and 185 raised the NLL from 1.6036 to 1.6102, i.e. +0.0066. That is above the 0.005 limit, but very noisy at this size. I'll repeat the same skeleton on the full 100 selected stays to get a reliable cost estimate before deciding whether this design is viable.

```python
print(json.dumps(sel))
```
The vocab lists the event kinds: 1 measurement, 2 drug_admin, 3 visit, 4 visit_detail. A wrong event-kind index in my measurement atoms is a plausible cause of the old 33% e_k error.

To build measurement and visit atoms explicitly, I need the embedding tables' names and sizes. That shows which index each lookup uses (kind, ekind, the categorical unit tables).

```python
for n in NAMES:
    if n.startswith('embed') or n.startswith('encode.'):
        print(n, SHAPES[n])
```
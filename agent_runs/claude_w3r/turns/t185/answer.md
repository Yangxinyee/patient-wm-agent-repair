Run 172 shows that impl4.py and impl2.py only hold helper code: the head re-implementation, bin and z-score helpers and the band policy. The core-step functions in the kernel have no saved source. They still work as black boxes. Runs ~118–170 are not documented, so I'll look at the kernel's string variables next. `notes_add`, `code`, `src` and `code_txt` may hold the most recent code or unsaved notes, and arrays of length 9918 are what run 170 used. If `notes_add` holds notes, I'll save them. If not, I'll work out what was done from the variable shapes.

```python
for k in ['notes_add','code','code_txt','src','src2','cc','ed','ev','head','kw','line','txt','scn','var','dm','et','f','fn_','g_','hk','kf','kk','l','n','n_','name','nm','nm_','nmx','p','pm','pv']:
    v=globals().get(k)
    if isinstance(v,str): print('##',k,len(v),repr(v[:300]))
print({k:(tuple(v.shape)) for k,v in globals().items() if hasattr(v,'shape') and len(getattr(v,'shape',()))>0 and v.shape[0] in (9918,)})
```
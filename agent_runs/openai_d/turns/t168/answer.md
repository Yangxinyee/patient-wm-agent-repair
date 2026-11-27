The architecture confirms that a value-head edit can leave the recurrent dynamics unchanged. We still need the record-based effect estimates and the candidate’s construction, rather than assuming any drug behavior. I’ll recover those next; they should tell us whether the priority is correcting effect size or protecting factual fit.

```python
_notes = Path("/work/notes.md").read_text()
print("=== RECORD-BASED EFFECT ESTIMATES ===")
_start = _notes.index("## Run 3\n")
_end = _notes.index("## Run 7\n")
print(_notes[_start:_end][:5200])

print("=== ACCEPTED INSULIN CANDIDATE AND STRENGTH TESTS ===")
_start = _notes.index("## Run 103:")
_end = _notes.index("## Run 124:")
print(_notes[_start:_end][:3600])

print("=== AVAILABLE RECENT VARIABLES ===")
_recent = [n for n in globals() if re.match(r"r(10[0-9]|11[0-9]|12[0-9]|13[0-9]|14[0-9])", n)]
print(sorted(_recent))
```

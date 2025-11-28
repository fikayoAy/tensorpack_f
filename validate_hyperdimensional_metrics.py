import json

# Load connections file which may be either a payload dict or a list of rows
p = json.load(open('connections_linked.json'))
if isinstance(p, list):
    # Normalize list-of-rows into a simple connections mapping keyed by src
    conns = {}
    for row in p:
        try:
            src = row.get('source_element_index')
        except Exception:
            src = None
        if src is None:
            continue
        # keep both int and str keys for flexible lookup
        conns.setdefault(src, []).append(row)
        conns.setdefault(str(src), []).append(row)
else:
    conns = p.get('connections', {}) if isinstance(p, dict) else {}

res = json.load(open('test_results.json'))
for entry in res:
    t = entry.get('tgt')
    has = (t in conns) or (str(t) in conns)
    # try integer form if t looks numeric
    try:
        if not has:
            has = int(t) in conns
    except Exception:
        pass
    print(t, 'has_outgoing:', bool(has))
import json

# Load connections
data = json.load(open('connections_linked.json'))

# Check both targets
targets = [
    'CID3586_AACCGCGCATGCTAGT',  # Target from test_directional.json (worked - 1 item)
    'CID3586_AAAGCAAGTGCAACGA'   # Target from test_directional_valid.json (26055 items)
]

for target in targets:
    matches = [e for e in data if target in e.get('target_row', '')]
    print(f'\nTarget: {target}')
    print(f'  Appears {len(matches)} times as target')
    if matches:
        first = matches[0]
        print(f'  First match: source_idx={first.get("source_element_index")}, target_idx={first.get("target_element_index")}')
        # Check the target row signature
        tgt_row = first.get('target_row', '')
        sig = tgt_row.split('\t')[0].strip().strip('"') if '\t' in tgt_row else tgt_row.strip('"')
        print(f'  Target signature: {sig}')

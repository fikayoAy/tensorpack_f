"""
Programmatic example: call traverse_graph_command from the tensorpack package

This script mirrors the discover example but allows choosing the exploration mode
(programmatic) at runtime via an environment variable or command-line arg.

Modes supported:
 - pathway: find a pathway between two datasets (requires source and target)
 - bridges: find bridge datasets (--find_bridges)
 - entity: search for a specific entity across datasets (--search_entity)

Adjust file paths and options as needed for your environment.
"""
import os
import sys
from argparse import Namespace
import logging
from pathlib import Path

# Add parent directory to path to import from tensorpack2
sys.path.insert(0, str(Path(__file__).parent.parent))

from tgc import traverse_graph_command


def build_args(mode='entity'):
    # Base args shared by all modes
    base = dict(
        inputs=['part-00520-8d7288b5-e7b4-445f-98f7-dd96cdd21fe0.c000.csv', 'part-00520-59bb15b1-8575-4586-8e15-e12f498bc0a6.c000.csv', 'part-00520-3c39916b-5ac3-4330-98a2-c4017d4667da.c000.csv'],
        apply_transform=None,
        verbose=True,
        log=None,
        save_npy=False,
        save_npz=False,
    )

    if mode == 'pathway':
        base.update({
            'source_dataset': 'examples/sample_1.npy',
            'target_dataset': 'examples/sample_2.csv',
            'find_bridges': False,
            'search_entity': None,
            'output': 'examples/pathway_results',
        })
    elif mode == 'entity':
        base.update({
            'search_entity': '48901',
            'find_bridges': False,
            'source_dataset': None,
            'target_dataset': None,
            'output': 'examples/entity_search_results',
        })
    else:  # default 'bridges'
        base.update({
            'find_bridges': True,
            'search_entity': None,
            'source_dataset': None,
            'target_dataset': None,
            'output': 'examples/bridge_results',
        })

    return Namespace(**base)


def main():
    # Dynamic mode selection: try command-line first, then env var, then default
    mode = None
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = os.environ.get('TP_MODE', 'bridges').lower()

    if mode not in ('pathway', 'bridges', 'entity'):
        print("Invalid mode. Choose one of: pathway, bridges, entity")
        sys.exit(2)

    args = build_args(mode)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    rc = traverse_graph_command(args)
    print(f"traverse_graph_command returned: {rc}")


if __name__ == '__main__':
    main()

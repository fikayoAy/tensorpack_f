"""Small runner to invoke tensorpack.dcc.discover_connections_command on sample files.

This script builds a minimal argparse.Namespace with the options used by the command
and calls the function directly (useful for testing without going through the CLI).
"""
import os
import argparse
from types import SimpleNamespace

# Make sure package import path is correct when running from repo root
# Example usage: python scripts/run_dcc_on_samples.py

from tensorpack import dcc

# Paths to datasets (absolute as provided by user)
SAMPLE_FILES = [
    r"part-00520-3c39916b-5ac39916b-5ac3-4330-98a2-c4017d4667da.c000.csv",
    r"part-00520-8d7288b5-e7b4-445f-98f7-dd96cdd21fe0.c000.csv",
    r"part-00520-59bb15b1-8575-4586-8e15-e12f498bc0a6.c000.csv",
    r"part-00520-77a06914-3875-421a-9622-699fe3a3b520.c000.json",
]


def build_args():
    # Mirror arguments expected by discover_connections_command
    return SimpleNamespace(
        inputs=SAMPLE_FILES,
        apply_transform=None,
        save_npy=False,
        num_dims=16,
        threshold=0.2,
        max_connections=10,
        clustering='auto',
        clustering_enabled=True,
        output='sample_connection_results.json',
        visualize=False,
        format='json',
        verbose=True,
        skip_errors=True,
        max_datasets=None
    )


if __name__ == '__main__':
    args = build_args()
    # Ensure files exist before calling
    missing = [p for p in args.inputs if not os.path.exists(p)]
    if missing:
        print('Missing sample files:')
        for m in missing:
            print('  ', m)
        print('Fix paths or place sample files in those locations before running.')
    else:
        rc = dcc.discover_connections_command(args)
        print('discover_connections_command returned:', rc)

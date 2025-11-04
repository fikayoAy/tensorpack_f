"""
Programmatic example: call discover_connections_command from the tensorpack package

This example shows how to construct an argparse.Namespace (or similar object)
with the same attributes the CLI expects and call the command function directly.

Adjust file paths and options as needed for your environment.
"""
from argparse import Namespace
import logging
import sys
from pathlib import Path

# Add parent directory to path to import from tensorpack2
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the command function from the package
from dcc import discover_connections_command


def main():
    # Build an args namespace with typical options used by the CLI
    args = Namespace(
        inputs=['part-00520-8d7288b5-e7b4-445f-98f7-dd96cdd21fe0.c000.csv', 'part-00520-59bb15b1-8575-4586-8e15-e12f498bc0a6.c000.csv'],
        apply_transform=None,
        num_dims=8,
        threshold=0.35,
        max_connections=10,
        clustering='auto',
        visualize=None,  # Set to a path string to request visual output (suppressed by config)
        export_formats=['json', 'csv'],
        output='examples/connections.json',
        format=None,
        verbose=True,
        log=None,
        skip_errors=False,
        save_npy=False,
        save_npz=False
    )

    # Optionally configure logging
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    # Call the command function directly
    rc = discover_connections_command(args)
    print(f"discover_connections_command returned: {rc}")


if __name__ == '__main__':
    main()

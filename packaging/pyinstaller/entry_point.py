"""PyInstaller entry point for srclight.

This thin wrapper avoids the 'relative import with no known parent package'
error that occurs when PyInstaller tries to run cli.py directly (which uses
``from . import __version__``).
"""

import multiprocessing

from srclight.cli import main

if __name__ == "__main__":
    # A frozen build starts its worker processes (the call graph scan) as
    # this same executable with `--multiprocessing-fork`: this turns them
    # into workers before the CLI can reject the flag. Nothing otherwise.
    multiprocessing.freeze_support()
    main()

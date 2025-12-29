"""
cFUSE Command Line Interface

Entry point for the cfuse-optimize command.
"""

def main():
    """Main entry point for cfuse-optimize command"""
    try:
        from optimize_basin import main as optimize_main
    except ModuleNotFoundError:
        # Fallback for repo checkout where optimize_basin.py sits in python/
        import sys
        from pathlib import Path

        _parent = Path(__file__).parent.parent
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        from optimize_basin import main as optimize_main
    optimize_main()


if __name__ == "__main__":
    main()

"""Compare compatible measured Bonsai decode JSON files; no hardware access."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opentpu.llm.bonsai_profile import compare


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('before', type=Path)
    ap.add_argument('after', type=Path)
    args = ap.parse_args()
    try:
        result = compare(json.loads(args.before.read_text()), json.loads(args.after.read_text()))
    except (OSError, ValueError, KeyError) as exc:
        ap.exit(2, f'cannot compare profiles: {exc}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

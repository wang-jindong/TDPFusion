"""Select an existing dataset trainer without changing its training protocol."""
import argparse
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    selector = argparse.ArgumentParser(description=__doc__, add_help=False)
    selector.add_argument('--dataset', choices=['msrs', 'llvip', 'roadscene'], required=True)
    selector.add_argument('--clip-path', default=str(ROOT / 'clip_weights'))
    if len(sys.argv) == 1 or (('--help' in sys.argv or '-h' in sys.argv) and '--dataset' not in sys.argv
                             and not any(x.startswith('--dataset=') for x in sys.argv)):
        selector.print_help()
        print('\nUse --dataset NAME --help for the original trainer options.')
        return
    selected, _ = selector.parse_known_args()
    module = importlib.import_module('training.' + selected.dataset)
    parser = module.build_parser()
    parser.description = f'Existing {selected.dataset} training implementation; see TRAINING.md for provenance.'
    parser.add_argument('--dataset', choices=['msrs', 'llvip', 'roadscene'], required=True)
    parser.add_argument('--clip-path', default=str(ROOT / 'clip_weights'))
    args = parser.parse_args()
    os.environ['TDPFUSION_CLIP_PATH'] = str(Path(args.clip_path).resolve())
    module.main(args)


if __name__ == '__main__':
    main()

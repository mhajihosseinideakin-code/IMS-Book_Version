"""
case.cli
--------

Command-line front door to the case-based interface:

    python -m ims_platform.case.cli list-models
    python -m ims_platform.case.cli template grid_forming_inverter my_case.yaml
    python -m ims_platform.case.cli run my_case.yaml
    python -m ims_platform.case.cli wizard
    python -m ims_platform.case.cli wizard --save my_case.yaml   (save without running)
"""

from __future__ import annotations

import argparse
import sys

from .loader import load_case, save_case_template
from .runner import CaseRunner
from .registry import list_models
from .wizard import run_wizard


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="ims_platform.case.cli",
        description="IMS Platform: enter your power system's data and get an automated "
                    "large-signal stability & recoverability analysis with plots and a report.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-models", help="List available system models and their parameters.")

    p_template = sub.add_parser("template", help="Write a starter case file for a given model.")
    p_template.add_argument("model", help="Model key, e.g. grid_forming_inverter")
    p_template.add_argument("path", help="Output path, e.g. my_case.yaml")

    p_run = sub.add_parser("run", help="Run the full analysis described by a case file.")
    p_run.add_argument("case_file", help="Path to a .yaml/.yml/.json case file.")
    p_run.add_argument("--quiet", action="store_true", help="Suppress progress logging.")

    p_wizard = sub.add_parser("wizard", help="Interactively enter your system's data.")
    p_wizard.add_argument("--save", metavar="PATH", default=None,
                           help="Save the case file to PATH instead of running it immediately.")
    p_wizard.add_argument("--quiet", action="store_true", help="Suppress progress logging when running.")

    args = parser.parse_args(argv)

    if args.command == "list-models":
        print(list_models())
        return 0

    if args.command == "template":
        save_case_template(args.model, args.path)
        print(f"Wrote starter case file to {args.path}. Edit it, then run:\n"
              f"  python -m ims_platform.case.cli run {args.path}")
        return 0

    if args.command == "run":
        case = load_case(args.case_file)
        runner = CaseRunner(case, verbose=not args.quiet)
        results = runner.run()
        print(f"\nAnalysis complete. Outputs written to: {results['output_dir']}")
        return 0

    if args.command == "wizard":
        case = run_wizard()
        if args.save:
            import os
            from .loader import _HAS_YAML  # noqa
            ext = os.path.splitext(args.save)[1].lower()
            import json
            with open(args.save, "w") as f:
                if ext in (".yaml", ".yml"):
                    import yaml
                    yaml.safe_dump(case.to_dict(), f, sort_keys=False)
                else:
                    json.dump(case.to_dict(), f, indent=2)
            print(f"Saved case file to {args.save}. Run it later with:\n"
                  f"  python -m ims_platform.case.cli run {args.save}")
        else:
            runner = CaseRunner(case, verbose=not args.quiet)
            results = runner.run()
            print(f"\nAnalysis complete. Outputs written to: {results['output_dir']}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

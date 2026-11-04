"""Integrated demo: input, solution, and plots of how search, NN and CSP cooperate. Owner: Person B.

Examples:
    python run_demo.py --profile example
    python run_demo.py --profile example --strategy greedy --csp bt --trace
    python run_demo.py --profile example --predictor avg
"""
import argparse


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", default="example", help="profile id, or 'example'")
    p.add_argument("--strategy", choices=["ucs", "greedy", "astar"], default="ucs")
    p.add_argument("--csp", choices=["bt", "fc", "ac3"], default="ac3",
                   help="bt = backtracking only, fc = +forward checking, ac3 = +FC +AC-3")
    p.add_argument("--predictor", choices=["nn", "avg"], default="nn")
    p.add_argument("--trace", action="store_true", help="print the search trace")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print(f"Demo settings: {vars(args)}")
    print("Planner not implemented yet (milestone M3, end of Week 5).")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
MatAgentBench - Agentic AI Evaluation Framework for Materials Science
======================================================================

This is a demonstration launcher for the MatAgentBench framework.
It evaluates LLM agents on autonomous materials science workflows.

Quick Start:
    python agenticai.py                  # Interactive menu
    python agenticai.py --demo           # Run a quick demo
    python agenticai.py --help           # Show all CLI commands

For full documentation, see README.md
"""

import sys
from pathlib import Path

# Add src to Python path so imports work
sys.path.insert(0, str(Path(__file__).parent / "src"))


def print_banner():
    """Display welcome banner."""
    print("\n" + "=" * 70)
    print("  MatAgentBench - Agentic AI Evaluation Framework")
    print("  Autonomous Materials Science Agents with Failure Attribution")
    print("=" * 70 + "\n")


def print_menu():
    """Display interactive menu."""
    print("What would you like to do?\n")
    print("  [1] Check environment & dependencies (mab doctor)")
    print("  [2] View project architecture & features")
    print("  [3] Open full CLI help")
    print("  [4] Exit")
    print()


def show_architecture():
    """Display project architecture overview."""
    print("\n" + "=" * 70)
    print("  PROJECT ARCHITECTURE")
    print("=" * 70 + "\n")

    print("📦 Core Components:\n")
    print("  1. Agent Runner (src/matagentbench/agent/)")
    print("     → Executes LLM agents with tool-calling protocols")
    print("     → Supports multiple backends (OpenAI-compatible, local models)")
    print()
    print("  2. Physics Simulation (src/matagentbench/sim/)")
    print("     → CHGNet ML potential for atomic relaxation")
    print("     → ASE integration for materials calculations")
    print()
    print("  3. Verification Engine (src/matagentbench/verify/)")
    print("     → Classifies agent failures into 15+ categories")
    print("     → Detects silent errors (units, basis, cell conventions)")
    print()
    print("  4. Failure Attribution (src/matagentbench/attribute/)")
    print("     → Counterfactual interventions (plan repair, context restore)")
    print("     → Pinpoints root causes of agent failures")
    print()
    print("  5. Web Dashboard (site/)")
    print("     → Interactive leaderboard & trajectory viewer")
    print("     → Zero-dependency static site (HTML/CSS/JS)")
    print()

    print("🎯 Key Features:\n")
    print("  ✓ 7 materials task families (energy, modulus, defects, etc.)")
    print("  ✓ Silent failure detection (wrong units/basis without errors)")
    print("  ✓ Mechanical convention reconciliation (no LLM judge)")
    print("  ✓ Contamination guards (detects memorized vs computed answers)")
    print("  ✓ Counterfactual debugging (what-if intervention analysis)")
    print("  ✓ Multi-model benchmarking with caching & resumability")
    print()

    print("📊 Pipeline Commands:\n")
    print("  mab doctor              → Check dependencies & API keys")
    print("  mab gen-tasks           → Generate calibrated task set")
    print("  mab run --preset <name> → Run agent trajectories")
    print("  mab attribute           → Analyze failure modes")
    print("  mab report              → Build leaderboard JSON")
    print()

    input("Press Enter to return to menu...")


def run_doctor():
    """Run environment check."""
    print("\n" + "=" * 70)
    print("  CHECKING ENVIRONMENT & DEPENDENCIES")
    print("=" * 70 + "\n")

    try:
        from matagentbench.cli import app
        import typer.testing

        runner = typer.testing.CliRunner()
        result = runner.invoke(app, ["doctor"])
        print(result.output)
    except ImportError as e:
        print(f"❌ Import Error: {e}\n")
        print("💡 To install dependencies:")
        print("   pip install -e .")
        print("   pip install -e \".[sim,data,dev]\"  # Full installation")
        print()

    input("Press Enter to return to menu...")


def show_cli_help():
    """Show full CLI help."""
    print("\n" + "=" * 70)
    print("  FULL CLI COMMANDS")
    print("=" * 70 + "\n")

    try:
        from matagentbench.cli import app
        import typer.testing

        runner = typer.testing.CliRunner()
        result = runner.invoke(app, ["--help"])
        print(result.output)
    except ImportError:
        print("❌ Package not installed. Run: pip install -e .")
        print()

    input("Press Enter to return to menu...")


def interactive_mode():
    """Run interactive menu."""
    print_banner()

    while True:
        print_menu()

        try:
            choice = input("Enter your choice [1-4]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! 👋\n")
            sys.exit(0)

        if choice == "1":
            run_doctor()
        elif choice == "2":
            show_architecture()
        elif choice == "3":
            show_cli_help()
        elif choice == "4":
            print("\nGoodbye! 👋\n")
            break
        else:
            print("\n⚠️  Invalid choice. Please enter 1, 2, 3, or 4.\n")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="MatAgentBench - Agentic AI Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agenticai.py              # Interactive menu
  python agenticai.py --demo       # Quick demo (same as interactive)
  python agenticai.py --doctor     # Check environment only

For the full CLI, use the 'mab' command:
  mab doctor
  mab gen-tasks --limit 10
  mab run --preset mid-open
  mab report
        """
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run interactive demo mode"
    )

    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Check environment and dependencies only"
    )

    args = parser.parse_args()

    if args.doctor:
        print_banner()
        run_doctor()
    elif args.demo or len(sys.argv) == 1:
        # Default to interactive mode if no args
        interactive_mode()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

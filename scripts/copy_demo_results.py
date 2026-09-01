"""
Quick script to copy demo results to the main results folder for GitHub Pages deployment.
Run this once to populate results/ with sample data for the web dashboard.
"""
import shutil
from pathlib import Path

demo_root = Path(".demo/results")
target_root = Path("results")

# Files already copied:
# - results/leaderboard.json
# - results/taskset.json
# - results/trajectories/index.json
# - results/trajectories/cerebras-gpt-oss-120b/formation_energy__mp-2657__seed0.json
# - results/trajectories/cerebras-gpt-oss-120b/density__mp-2657__seed0.json

# Copy remaining trajectory files
remaining_trajectories = [
    "trajectories/cerebras-gpt-oss-120b/vacancy_formation__mp-2657__seed0.json",
    "trajectories/groq-qwen-qwen3-32b/formation_energy__mp-2657__seed0.json",
    "trajectories/groq-qwen-qwen3-32b/density__mp-2657__seed0.json",
    "trajectories/groq-qwen-qwen3-32b/vacancy_formation__mp-2657__seed0.json",
    "trajectories/groq-qwen-qwen3-32b/bulk_modulus__mp-2657__seed0.json",
]

# Copy counterfactual files
counterfactual_files = [
    "counterfactuals/cerebras-gpt-oss-120b/formation_energy__mp-2657__convention_correct__k3.json",
]

for rel_path in remaining_trajectories + counterfactual_files:
    source = demo_root / rel_path
    target = target_root / rel_path

    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        print(f"✓ Copied {rel_path}")
    else:
        print(f"⚠ Skipped {rel_path} (not found)")

print(f"\n✅ Demo results ready in results/")
print(f"   Site will render at: site/index.html")

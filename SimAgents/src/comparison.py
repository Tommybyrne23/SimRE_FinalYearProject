"""
comparison.py - Comparison Report Generator for SimAgents Project

Reads existing evaluation.json files from all completed run directories
and generates a side-by-side comparison report. No ML models are loaded —
this script only reads and formats already-computed evaluation results.

Run:
    python src/comparison.py outputs/

Outputs:
    outputs/comparison_report.txt  — human-readable comparison table
    outputs/comparison.json        — machine-readable summary of all runs
"""

import json
import argparse
import sys
from pathlib import Path
from datetime import datetime


SCENARIO_ORDER = [
    'barber_booking',
    'clinic_patient_portal',
    'library_system',
    'student_portal_enhancement',
    'banking_app_modernisation',
]

SCENARIO_LABELS = {
    'barber_booking':             'Barbershop Booking System',
    'clinic_patient_portal':      'GP Clinic Patient Portal',
    'library_system':             'University Library System',
    'student_portal_enhancement': 'Student Portal Enhancement',
    'banking_app_modernisation':  'Banking App Modernisation',
}

PERSONA_SHORT = {
    'experienced_methodical': 'Senior (David)',
    'adaptive_generalist':    'Mid-level (Eric)',
    'junior_inexperienced':   'Junior (Rachel)',
}

USER_SHORT = {
    'cooperative_expert':         'Cooperative Expert',
    'vague_novice':               'Vague Novice',
    'resistant_expert':           'Resistant Expert',
    'clinic_cooperative_expert':  'Cooperative Expert',
    'clinic_vague_novice':        'Vague Novice',
    'clinic_resistant_expert':    'Resistant Expert',
    'library_cooperative_expert': 'Cooperative Expert',
    'library_vague_novice':       'Vague Novice',
    'library_resistant_expert':   'Resistant Expert',
    'portal_cooperative_expert':  'Cooperative Expert',
    'portal_vague_novice':        'Vague Novice',
    'portal_resistant_expert':    'Resistant Expert',
    'banking_cooperative_expert': 'Cooperative Expert',
    'banking_vague_novice':       'Vague Novice',
    'banking_resistant_expert':   'Resistant Expert',
}

TYPE_DISPLAY = {
    'FunctionalRequirement':    'Functional',
    'NonFunctionalRequirement': 'Non-Functional',
    'Constraint':               'Constraint',
}


def _infer_labels(run_id: str) -> tuple:
    """Extract short RE persona and user persona labels from a run_id string."""
    re_label   = next((v for k, v in PERSONA_SHORT.items() if k in run_id), run_id)
    user_label = next((v for k, v in USER_SHORT.items()    if k in run_id), run_id)
    return re_label, user_label


def _scenario_for(run_id: str) -> str:
    """Infer scenario ID from run_id string."""
    return next((s for s in SCENARIO_ORDER if s in run_id), 'unknown')


def _fmt(val) -> str:
    """Format a metric value for display."""
    return f"{val:.4f}" if val is not None else "N/A"


def load_all_evaluations(outputs_dir: Path) -> list:
    """
    Load all evaluation.json files from run subdirectories.

    Args:
        outputs_dir (Path): Path to the outputs directory.

    Returns:
        list of evaluation result dicts, sorted by run_id.
    """
    results = []
    run_dirs = sorted([d for d in outputs_dir.iterdir()
                       if d.is_dir() and d.name.startswith('run_')])

    if not run_dirs:
        print(f"[ERROR] No run directories found in {outputs_dir}")
        sys.exit(1)

    for run_dir in run_dirs:
        eval_file = run_dir / "evaluation.json"
        if not eval_file.exists():
            print(f"  [SKIP] {run_dir.name} — evaluation.json not found. Run evaluation.py first.")
            continue
        with open(eval_file, 'r', encoding="utf-8") as f:
            results.append(json.load(f))

    if not results:
        print("[ERROR] No evaluation.json files found. Run evaluation.py on your runs first.")
        sys.exit(1)

    print(f"  Loaded {len(results)} evaluation(s).\n")
    return results


def generate_report(all_results: list) -> list:
    """
    Build the comparison report as a list of text lines.

    Groups runs by scenario, then prints a metrics table and a
    coverage breakdown table for each scenario.

    Args:
        all_results (list): List of evaluation result dicts.

    Returns:
        list of strings — one per line of the report.
    """
    col_w = 20

    # Group results by scenario
    grouped = {}
    for r in all_results:
        scenario_id = _scenario_for(r['run_id'])
        grouped.setdefault(scenario_id, []).append(r)

    lines = []
    lines.append("=" * 80)
    lines.append("  SIMAGENTS EVALUATION REPORT")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 80)

    for scenario_id in SCENARIO_ORDER:
        runs = grouped.get(scenario_id, [])
        if not runs:
            continue

        label = SCENARIO_LABELS.get(scenario_id, scenario_id)
        lines.append(f"\n  {label}")
        lines.append("  " + "─" * 76)

        # Metrics table 
        metric_keys   = ['coherence_mean', 're_informativeness_mean',
                         'user_informativeness_mean', 'requirement_coverage',
                         'requirement_specificity_mean']
        metric_labels = ['Coherence', 'RE Info', 'User Info', 'Coverage', 'Specificity']

        header = (f"  {'RE Persona':<{col_w}}  {'User Persona':<{col_w}}" +
                  "".join(f"  {lbl:>{col_w}}" for lbl in metric_labels))
        lines.append(header)
        lines.append("  " + "─" * (col_w * 2 + col_w * len(metric_labels) + 2 * (len(metric_labels) + 2)))

        for r in sorted(runs, key=lambda x: x['run_id']):
            re_lbl, user_lbl = _infer_labels(r['run_id'])
            row = f"  {re_lbl:<{col_w}}  {user_lbl:<{col_w}}"
            for key in metric_keys:
                val = r['summary'].get(key)
                row += f"  {_fmt(val):>{col_w}}"
            lines.append(row)

        # Coverage breakdown 
        lines.append("")
        lines.append(f"  {'Coverage Breakdown'}")
        type_keys   = ['FunctionalRequirement', 'NonFunctionalRequirement', 'Constraint']
        type_labels = [TYPE_DISPLAY[t] for t in type_keys]

        bk_header = (f"  {'RE Persona':<{col_w}}  {'User Persona':<{col_w}}" +
                     "".join(f"  {lbl:>{col_w}}" for lbl in type_labels))
        lines.append(bk_header)
        lines.append("  " + "─" * (col_w * 2 + col_w * len(type_keys) + 2 * (len(type_keys) + 2)))

        for r in sorted(runs, key=lambda x: x['run_id']):
            re_lbl, user_lbl = _infer_labels(r['run_id'])
            by_type = r['metrics']['requirement_coverage'].get('by_type', {})
            row = f"  {re_lbl:<{col_w}}  {user_lbl:<{col_w}}"
            for t in type_keys:
                row += f"  {_fmt(by_type.get(t)):>{col_w}}"
            lines.append(row)

    lines.append(f"\n{'='*80}\n")
    return lines


# Entry Point 
def compare_runs(outputs_dir: Path):
    """
    Load all evaluations, generate the comparison report, and save outputs.

    Args:
        outputs_dir (Path): Path to the outputs directory containing run subdirectories.
    """
    print(f"\n{'='*60}")
    print("SimAgents Comparison Report")
    print(f"{'='*60}\n")

    all_results = load_all_evaluations(outputs_dir)
    lines       = generate_report(all_results)

    # Print to terminal
    for line in lines:
        print(line)

    # Save plain text report
    report_file = outputs_dir / "comparison_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"  Report saved to: {report_file}")

    # Save comparison JSON
    comparison_data = {
        'compared_at':   datetime.now().isoformat(),
        'runs':          [r['run_id'] for r in all_results],
        'summary_table': {r['run_id']: r['summary'] for r in all_results}
    }
    comp_file = outputs_dir / "comparison.json"
    with open(comp_file, 'w', encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)
    print(f"  Comparison JSON saved to: {comp_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SimAgents Comparison — generates a comparison report from completed run evaluations"
    )
    parser.add_argument('outputs_dir', help="Path to the outputs directory containing run subdirectories")
    args = parser.parse_args()

    target = Path(args.outputs_dir)
    if not target.is_dir():
        print(f"[ERROR] {target} is not a directory.")
        sys.exit(1)

    compare_runs(target)

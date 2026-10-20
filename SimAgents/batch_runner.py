"""
batch_runner.py - Run All SimAgents Configurations

Orchestrates execution of the full 5 x 3 x 3 = 45 configuration matrix:
    5 scenarios x 3 RE personas x 3 user archetypes per scenario.

Per configuration the pipeline runs:
    1. SimAgentsSession   - run dialogue
    2. extract_requirements_llm - extract requirements
    3. evaluate_run       - compute metrics  (skipped if eval deps missing or --no-eval)

After all configurations the script aggregates results via comparison.py logic
to produce outputs/comparison_report.txt and outputs/comparison.json.

Features:
    - Resume: skips configurations whose dialogue.json already exists (use --force to re-run)
    - Per-config error isolation: one failure does not stop the batch
    - Per-config timing and a final batch_log.json summary
    - CLI flags for partial runs, dry runs, evaluation toggle

Usage:
    python batch_runner.py                       # run all 45 (or what's missing)
    python batch_runner.py --force               # re-run all 45 from scratch
    python batch_runner.py --batch 2             # second batch (appends _batch2 to run IDs)
    python batch_runner.py --batch 3             # third batch (appends _batch3 to run IDs)
    python batch_runner.py --scenarios 0,1       # only first two scenarios
    python batch_runner.py --re-personas 0       # only senior RE
    python batch_runner.py --archetypes 1,2      # only vague + resistant
    python batch_runner.py --no-eval             # skip evaluation step
    python batch_runner.py --dry-run             # list configs without running
"""

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Allow imports from src/ (matches main.py and server.py conventions)
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents     import create_model_client, create_re_agent, create_user_agent
from session    import SimAgentsSession
from extraction import extract_requirements_llm

# Evaluation is optional - degrade gracefully if deps missing
try:
    from evaluation import evaluate_run
    EVAL_AVAILABLE = True
except Exception as e:
    EVAL_AVAILABLE      = False
    EVAL_IMPORT_ERROR   = str(e)

# Comparison report is local file aggregation - always available
try:
    from comparison import compare_runs
    COMPARE_AVAILABLE = True
except Exception:
    COMPARE_AVAILABLE = False


CONFIG_DIR  = Path("config")
OUTPUTS_DIR = Path("outputs")

# Archetype offsets within each scenario's user persona group.
# Per main.py convention: each scenario has 3 user personas in this order.
ARCHETYPE_NAMES = ['cooperative_expert', 'vague_novice', 'resistant_expert']


# Config Loading
def load_configs():
    """Load personas.json and scenarios.json from CONFIG_DIR."""
    with open(CONFIG_DIR / "personas.json", encoding="utf-8") as f:
        personas = json.load(f)
    with open(CONFIG_DIR / "scenarios.json", encoding="utf-8") as f:
        scenarios = json.load(f)
    return personas, scenarios


def build_matrix(personas, scenarios, scenario_filter, re_filter, archetype_filter, batch_suffix=None):
    """
    Build the list of configurations to run.

    Each user persona is grouped by scenario in personas.json - persona index
    is computed as scenario_idx * 3 + archetype_idx.

    Args:
        personas:           personas dict from personas.json
        scenarios:          scenarios dict from scenarios.json
        scenario_filter:    set of scenario indices to include (or None for all)
        re_filter:          set of RE persona indices (or None for all)
        archetype_filter:   set of archetype indices 0/1/2 (or None for all)
        batch_suffix:       optional string appended to each run_id (e.g., '_batch2')

    Returns:
        list of dicts: each with scenario, re_persona, user_persona, run_id
    """
    re_personas   = personas['re_personas']
    user_personas = personas['user_personas']

    matrix = []
    for s_idx, scenario in enumerate(scenarios['scenarios']):
        if scenario_filter is not None and s_idx not in scenario_filter:
            continue
        for r_idx, re_persona in enumerate(re_personas):
            if re_filter is not None and r_idx not in re_filter:
                continue
            for a_idx, archetype_name in enumerate(ARCHETYPE_NAMES):
                if archetype_filter is not None and a_idx not in archetype_filter:
                    continue

                # Look up the user persona for this scenario+archetype combo.
                # Match by scenario id appearing in persona id (or bare archetype
                # name for the first scenario, which uses unprefixed ids).
                user_persona = _lookup_user_persona(user_personas, scenario['id'], archetype_name)
                if user_persona is None:
                    print(f"  [WARN] No user persona found for {scenario['id']} + {archetype_name}")
                    continue

                run_id = f"run_{scenario['id']}_{re_persona['id']}_vs_{user_persona['id']}"
                if batch_suffix:
                    run_id += batch_suffix
                matrix.append({
                    'scenario':     scenario,
                    're_persona':   re_persona,
                    'user_persona': user_persona,
                    'run_id':       run_id,
                })
    return matrix


def _lookup_user_persona(user_personas, scenario_id, archetype_name):
    """
    Find the user persona matching a scenario+archetype.

    Personas use either:
        - bare archetype name (e.g., 'cooperative_expert')        - scenario 1
        - scenario_prefix_archetype (e.g., 'clinic_vague_novice') - scenarios 2-5

    Returns the matching persona dict or None if not found.
    """
    # First scenario uses bare archetype names; others use prefixed.
    candidates = [
        archetype_name,                          # bare (barber_booking)
        f"clinic_{archetype_name}",
        f"library_{archetype_name}",
        f"portal_{archetype_name}",
        f"banking_{archetype_name}",
    ]
    # Match against scenario by checking persona's 'scenario' field if present
    for p in user_personas:
        if p['id'] in candidates and p.get('scenario', '') in (scenario_id, ''):
            # Prefer exact scenario match if persona has scenario field
            if p.get('scenario') == scenario_id:
                return p
    # Fallback: ignore scenario field, match purely on id structure
    for p in user_personas:
        if p['id'] == archetype_name and scenario_id == 'barber_booking':
            return p
        if p['id'].startswith(scenario_id.split('_')[0]) and p['id'].endswith(archetype_name):
            return p
    return None


# Single-config Runner
async def run_one_config(cfg, llm_settings, run_eval):
    """
    Execute one full configuration: session -> extraction -> evaluation.

    Args:
        cfg:           dict from build_matrix
        llm_settings:  dict with use_local, api_key, model_name, temperature, seed
        run_eval:      bool - whether to run evaluation after extraction

    Returns:
        dict with keys: status ('success'|'error'), error (if any),
                        n_requirements, eval_summary, duration_s
    """
    start = time.time()
    result = {
        'run_id':         cfg['run_id'],
        'scenario':       cfg['scenario']['name'],
        're_persona':     cfg['re_persona']['name'],
        'user_persona':   cfg['user_persona']['name'],
        'status':         'unknown',
        'error':          None,
        'n_requirements': 0,
        'eval_summary':   None,
        'duration_s':     0.0,
    }

    try:
        # Build clients (one per agent so they don't share state)
        re_model   = create_model_client(**llm_settings)
        user_model = create_model_client(**llm_settings)

        re_agent   = create_re_agent  (re_model,   cfg['scenario'], cfg['re_persona'])
        user_agent = create_user_agent(user_model, cfg['user_persona'], cfg['scenario'])

        session = SimAgentsSession(
            re_agent=re_agent,
            user_agents=[user_agent],
            scenario=cfg['scenario'],
            run_id=cfg['run_id'],
            model_name=llm_settings['model_name']
        )
        await session.run_session()

        # Extract requirements — reuse an existing client instead of creating a new one
        dialogue_file = session.output_dir / "dialogue.json"
        if dialogue_file.exists():
            extraction_client = create_model_client(**llm_settings)
            requirements = await extract_requirements_llm(dialogue_file, model_client=extraction_client)
            result['n_requirements'] = len(requirements)

        # Evaluate
        if run_eval and EVAL_AVAILABLE:
            try:
                eval_result = evaluate_run(session.output_dir)
                result['eval_summary'] = eval_result.get('summary', {})
            except Exception as e:
                result['error'] = f"Eval failed: {e}"
                # Don't fail the whole config - dialogue + extraction succeeded

        result['status'] = 'success'

    except Exception as e:
        result['status'] = 'error'
        result['error']  = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    result['duration_s'] = round(time.time() - start, 1)
    return result


# Batch Orchestrator
async def run_batch(matrix, force, run_eval):
    """
    Execute all configurations in matrix sequentially.

    Returns:
        list of result dicts from run_one_config.
    """
    load_dotenv()
    llm_settings = {
        'use_local':   os.getenv("USE_LOCAL", "true").lower() == "true",
        'api_key':     os.getenv("OPENAI_API_KEY"),
        'model_name':  os.getenv("LOCAL_MODEL", "llama3.2") if os.getenv("USE_LOCAL", "true").lower() == "true" else os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        'temperature': float(os.getenv("TEMPERATURE", "0.7")),
        'seed':        int(os.getenv("SEED")) if os.getenv("SEED") else None,
    }

    results = []
    total   = len(matrix)
    skipped = 0
    succeeded = 0
    failed    = 0

    print("\n" + "=" * 70)
    print(f"  SimAgents Batch Runner")
    print(f"  Configurations: {total}")
    print(f"  Model:          {llm_settings['model_name']} ({'local' if llm_settings['use_local'] else 'OpenAI'})")
    print(f"  Evaluation:     {'on' if run_eval and EVAL_AVAILABLE else 'off'}")
    print(f"  Resume mode:    {'off (--force)' if force else 'on (skipping completed)'}")
    print("=" * 70 + "\n")

    for i, cfg in enumerate(matrix, start=1):
        run_dir   = OUTPUTS_DIR / cfg['run_id']
        dialogue_file = run_dir / "dialogue.json"
        prefix = f"[{i:>2}/{total}]"

        # Resume check
        if not force and dialogue_file.exists():
            print(f"{prefix} {cfg['run_id']:<70}  [SKIP - already completed]")
            results.append({
                'run_id': cfg['run_id'],
                'status': 'skipped',
                'error':  None,
                'duration_s': 0.0,
            })
            skipped += 1
            continue

        print(f"{prefix} {cfg['run_id']:<70}  ...", flush=True)
        result = await run_one_config(cfg, llm_settings, run_eval)
        results.append(result)

        # Status line
        if result['status'] == 'success':
            extras = f"{result['n_requirements']} reqs, {result['duration_s']}s"
            if result.get('eval_summary'):
                cov = result['eval_summary'].get('requirement_coverage')
                if cov is not None:
                    extras += f", cov={cov:.2f}"
            print(f"           ✓ {extras}")
            succeeded += 1
        else:
            print(f"           ✗ {result['error']}")
            failed += 1

    # Final summary
    print("\n" + "=" * 70)
    print(f"  Batch complete: {succeeded} succeeded, {failed} failed, {skipped} skipped")
    print("=" * 70 + "\n")

    return results


# Reporting
def save_batch_log(results, batch_suffix=None):
    """Persist batch results to outputs/batch_log.json for later inspection."""
    OUTPUTS_DIR.mkdir(exist_ok=True)
    log = {
        'completed_at': datetime.now().isoformat(),
        'batch':        batch_suffix or 'default',
        'total':        len(results),
        'succeeded':    sum(1 for r in results if r['status'] == 'success'),
        'failed':       sum(1 for r in results if r['status'] == 'error'),
        'skipped':      sum(1 for r in results if r['status'] == 'skipped'),
        'results':      results,
    }
    log_name = f"batch_log{batch_suffix or ''}.json"
    log_file = OUTPUTS_DIR / log_name
    with open(log_file, 'w', encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"  Batch log:        {log_file}")


def run_comparison():
    """Run comparison.py over all completed runs."""
    if not COMPARE_AVAILABLE:
        print("  [WARN] comparison.py not importable - skipping comparison report")
        return
    if not OUTPUTS_DIR.exists():
        print("  [WARN] outputs/ does not exist - skipping comparison report")
        return
    try:
        compare_runs(OUTPUTS_DIR)
    except SystemExit:
        # comparison.py calls sys.exit on no runs - suppress for batch context
        print("  [WARN] comparison.py exited - likely no runs to compare yet")
    except Exception as e:
        print(f"  [WARN] Comparison failed: {e}")


# CLI Entry Point
def parse_int_list(s):
    """Parse a comma-separated list of ints, e.g., '0,1,3' -> {0, 1, 3}."""
    if not s:
        return None
    return set(int(x.strip()) for x in s.split(',') if x.strip())


def main():
    parser = argparse.ArgumentParser(
        description="Run all 45 SimAgents configurations sequentially with resume support."
    )
    parser.add_argument('--force',       action='store_true',
                        help="Re-run configurations even if dialogue.json already exists")
    parser.add_argument('--no-eval',     action='store_true',
                        help="Skip the evaluation step after each session")
    parser.add_argument('--no-compare',  action='store_true',
                        help="Skip the final comparison report aggregation")
    parser.add_argument('--scenarios',   type=str, default=None,
                        help="Comma-separated scenario indices to include (0-4), e.g., '0,1'")
    parser.add_argument('--re-personas', type=str, default=None,
                        help="Comma-separated RE persona indices to include (0-2)")
    parser.add_argument('--archetypes',  type=str, default=None,
                        help="Comma-separated archetype indices to include (0=cooperative, 1=vague, 2=resistant)")
    parser.add_argument('--dry-run',     action='store_true',
                        help="List the configurations that would run, then exit")
    parser.add_argument('--batch',       type=int, default=None,
                        help="Batch number (e.g., 2 or 3). Appends '_batch<N>' to each run_id so multiple "
                             "batches coexist in outputs/ without overwriting. Omit for the first batch.")
    args = parser.parse_args()

    personas, scenarios = load_configs()
    batch_suffix = f"_batch{args.batch}" if args.batch else None
    matrix = build_matrix(
        personas,
        scenarios,
        parse_int_list(args.scenarios),
        parse_int_list(args.re_personas),
        parse_int_list(args.archetypes),
        batch_suffix=batch_suffix,
    )

    if args.dry_run:
        print(f"\nWould run {len(matrix)} configurations:")
        for i, cfg in enumerate(matrix, start=1):
            print(f"  [{i:>2}] {cfg['run_id']}")
        return

    if not matrix:
        print("[ERROR] No configurations matched the supplied filters.")
        sys.exit(1)

    if not EVAL_AVAILABLE and not args.no_eval:
        print(f"[WARN] Evaluation deps not available: {EVAL_IMPORT_ERROR}")
        print(f"[WARN] Continuing with evaluation disabled. Use --no-eval to silence this.")

    run_eval = (not args.no_eval) and EVAL_AVAILABLE

    results = asyncio.run(run_batch(matrix, args.force, run_eval))
    save_batch_log(results, batch_suffix)

    if not args.no_compare:
        print("\n  Running comparison aggregation...")
        run_comparison()


if __name__ == "__main__":
    main()

"""
evaluation.py - Automatic Evaluation Module for SimAgents Project

Computes five metrics against a single completed SimAgents run directory.
Results are saved to evaluation.json in the run directory and printed
to the terminal as a summary table.

Run:
    python src/evaluation.py outputs/run_barber_booking_experienced_methodical_vs_cooperative_expert/

Metrics:
    Dialogue Quality:
        1. Coherence             - semantic similarity between consecutive turns (Sentence-BERT)
        2. RE Informativeness    - noun phrase density of RE agent turns (spaCy)
        3. User Informativeness  - noun phrase density of User agent turns (spaCy)

    Requirements Quality:
        4. Requirement Coverage  - fraction of ground truth requirements semantically addressed (Sentence-BERT)
        5. Requirement Specificity - noun phrase density of extracted requirements (spaCy)

Dependencies:
    pip install sentence-transformers spacy
    python -m spacy download en_core_web_sm
"""

import json
import argparse
import sys
import os
import logging
from pathlib import Path
from datetime import datetime

# Suppress HuggingFace and transformer warnings
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)


# Model Loading
def _load_sbert():
    try:
        import transformers
        transformers.logging.set_verbosity_error()
        from sentence_transformers import SentenceTransformer, util
        return SentenceTransformer('all-MiniLM-L6-v2'), util
    except ImportError:
        print("[ERROR] sentence-transformers not installed. Run: pip install sentence-transformers")
        sys.exit(1)


def _load_spacy():
    try:
        import spacy
        return spacy.load('en_core_web_sm')
    except ImportError:
        print("[ERROR] spaCy not installed. Run: pip install spacy")
        sys.exit(1)
    except OSError:
        print("[ERROR] spaCy model not found. Run: python -m spacy download en_core_web_sm")
        sys.exit(1)


# Data Loading
def load_run(run_dir: Path) -> dict:
    """
    Load dialogue and requirements data from a completed run directory.

    Args:
        run_dir (Path): Path to run output directory containing dialogue.json.
            requirements.json is loaded if present, otherwise specificity is skipped.

    Returns:
        dict with keys: dialogue, requirements, run_id, run_dir
    """
    dialogue_file    = run_dir / "dialogue.json"
    requirements_file = run_dir / "requirements.json"

    if not dialogue_file.exists():
        raise FileNotFoundError(f"dialogue.json not found in {run_dir}")

    with open(dialogue_file, 'r', encoding="utf-8") as f:
        dialogue = json.load(f)

    requirements = None
    if requirements_file.exists():
        with open(requirements_file, 'r', encoding="utf-8") as f:
            requirements = json.load(f)

    return {
        'dialogue':     dialogue,
        'requirements': requirements,
        'run_id':       dialogue.get('run_id', run_dir.name),
        'run_dir':      run_dir
    }


def _load_scenario(run_id: str) -> dict | None:
    """
    Infer the scenario from the run_id and load it from config/scenarios.json.
    Returns None if no matching scenario is found.
    """
    scenarios_file = Path("config/scenarios.json")
    if not scenarios_file.exists():
        return None
    try:
        with open(scenarios_file, 'r', encoding="utf-8") as f:
            data = json.load(f)
        for scenario in data.get('scenarios', []):
            if scenario['id'] in run_id:
                return scenario
        return None
    except Exception:
        return None


def _split_messages(dialogue: dict) -> tuple:
    """
    Split dialogue messages into RE turns and User turns.
    Skips the system seed message (source='user').

    Returns:
        (re_turns, user_turns) — two lists of message content strings
    """
    re_turns   = []
    user_turns = []

    for msg in dialogue.get('messages', []):
        source  = msg.get('source', '')
        content = msg.get('content', '').strip()
        if not content:
            continue
        if source.startswith('RE_'):
            re_turns.append(content)
        elif source.startswith('User_'):
            user_turns.append(content)

    return re_turns, user_turns


# Coherence
def compute_coherence(dialogue: dict, sbert_model, sbert_util) -> dict:
    """
    Compute dialogue coherence as mean cosine similarity between consecutive turns.

    A high score indicates agents are responding meaningfully to each other.
    Uses Sentence-BERT (all-MiniLM-L6-v2) to encode turns into a semantic space.

    Score interpretation:
        0.0 - 0.3 : Low      — turns are largely unrelated
        0.3 - 0.6 : Moderate — some topical connection
        0.6 - 1.0 : High     — turns are semantically consistent
    """
    all_turns = [
        msg.get('content', '').strip()
        for msg in dialogue.get('messages', [])
        if msg.get('source', '') != 'user' and msg.get('content', '').strip()
    ]

    if len(all_turns) < 2:
        return {'mean': 0.0, 'std': 0.0, 'pairs_evaluated': 0,
                'scores': [], 'interpretation': 'Insufficient turns'}

    embeddings = sbert_model.encode(all_turns, convert_to_tensor=True)
    scores = [
        round(sbert_util.cos_sim(embeddings[i], embeddings[i + 1]).item(), 4)
        for i in range(len(embeddings) - 1)
    ]

    mean = round(sum(scores) / len(scores), 4)
    std  = round((sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5, 4)

    if mean >= 0.6:   interp = "High — turns are semantically consistent"
    elif mean >= 0.3: interp = "Moderate — some topical connection between turns"
    else:             interp = "Low — turns are largely unrelated"

    return {'mean': mean, 'std': std, 'pairs_evaluated': len(scores),
            'scores': scores, 'interpretation': interp}


# Informativeness
def compute_informativeness(turns: list, nlp) -> dict:
    """
    Compute informativeness as mean noun phrase density across a list of turns.

    Noun phrase density (NP count / token count) is used as a proxy for
    information richness. Higher density indicates more concrete, content-rich turns.

    Score interpretation:
        < 0.10       : Low      — brief or vague turns
        0.10 - 0.25  : Moderate — typical conversational density
        > 0.25       : High     — dense, content-rich turns
    """
    if not turns:
        return {'mean': 0.0, 'std': 0.0, 'turns_evaluated': 0,
                'per_turn': [], 'interpretation': 'No turns to evaluate'}

    densities = []
    for turn in turns:
        doc         = nlp(turn)
        token_count = len([t for t in doc if not t.is_space])
        np_count    = len(list(doc.noun_chunks))
        densities.append(round(np_count / token_count, 4) if token_count else 0.0)

    mean = round(sum(densities) / len(densities), 4)
    std  = round((sum((d - mean) ** 2 for d in densities) / len(densities)) ** 0.5, 4)

    if mean >= 0.25:   interp = "High — dense, content-rich turns"
    elif mean >= 0.10: interp = "Moderate — typical conversational density"
    else:              interp = "Low — brief or vague turns"

    return {'mean': mean, 'std': std, 'turns_evaluated': len(densities),
            'per_turn': densities, 'interpretation': interp}


# Requirement Coverage
def compute_coverage(dialogue: dict, scenario: dict,
                     sbert_model, sbert_util,
                     similarity_threshold: float = 0.4) -> dict:
    """
    Compute requirement coverage using SBERT semantic similarity.

    For each ground truth requirement, finds the maximum cosine similarity
    across all dialogue turns. A requirement is considered covered if this
    maximum exceeds the similarity_threshold. Results are broken down by
    requirement type (FunctionalRequirement, NonFunctionalRequirement, Constraint).

    Threshold of 0.4 reliably distinguishes topically related turns from
    coincidental keyword overlap while tolerating natural paraphrase variation.

    Score interpretation:
        < 0.5  : Low     — significant requirements were not elicited
        0.5-0.8: Partial — most areas addressed but some missed
        > 0.8  : High    — nearly all ground truth requirements addressed
    """
    ground_truth = scenario.get('ground_truth_requirements', [])

    if not ground_truth:
        return {'score': 0.0, 'covered': [], 'not_covered': [],
                'total_expected': 0, 'by_type': {},
                'similarity_threshold': similarity_threshold,
                'interpretation': 'No ground truth requirements defined in scenario'}

    turns = [
        msg.get('content', '').strip()
        for msg in dialogue.get('messages', [])
        if msg.get('source', '') != 'user' and msg.get('content', '').strip()
    ]

    if not turns:
        return {'score': 0.0, 'covered': [], 'not_covered': ground_truth,
                'total_expected': len(ground_truth), 'by_type': {},
                'similarity_threshold': similarity_threshold,
                'interpretation': 'No dialogue turns found'}

    turn_embeddings = sbert_model.encode(turns, convert_to_tensor=True)

    by_type     = {
        'FunctionalRequirement':    {'covered': 0, 'total': 0},
        'NonFunctionalRequirement': {'covered': 0, 'total': 0},
        'Constraint':               {'covered': 0, 'total': 0}
    }
    covered     = []
    not_covered = []

    for req in ground_truth:
        req_id        = req.get('id', '?')
        req_type      = req.get('type', 'FunctionalRequirement')
        req_statement = req.get('statement', '')

        if req_type in by_type:
            by_type[req_type]['total'] += 1

        req_embedding = sbert_model.encode(req_statement, convert_to_tensor=True)
        similarities  = sbert_util.cos_sim(req_embedding, turn_embeddings)[0]
        max_sim       = round(float(similarities.max().item()), 4)

        entry = {'id': req_id, 'type': req_type,
                 'statement': req_statement, 'max_similarity': max_sim}

        if max_sim >= similarity_threshold:
            covered.append(entry)
            if req_type in by_type:
                by_type[req_type]['covered'] += 1
        else:
            not_covered.append(entry)

    total      = len(ground_truth)
    score      = round(len(covered) / total, 4) if total else 0.0
    type_scores = {
        t: round(c['covered'] / c['total'], 4)
        for t, c in by_type.items() if c['total'] > 0
    }

    if score >= 0.8:   interp = "High — nearly all ground truth requirements addressed"
    elif score >= 0.5: interp = "Partial — most areas addressed but some missed"
    else:              interp = "Low — significant requirements were not elicited"

    return {
        'score': score, 'covered': covered, 'not_covered': not_covered,
        'total_expected': total, 'by_type': type_scores,
        'similarity_threshold': similarity_threshold, 'interpretation': interp
    }


# Requirement Specificity
def compute_specificity(requirements: dict, nlp) -> dict:
    """
    Compute requirement specificity as mean noun phrase density of extracted
    requirement statements.

    Higher NP density correlates with more concrete, testable requirements.
    Requires requirements.json to have been generated by extraction.py first.

    Score interpretation:
        < 0.15       : Low      — vague requirements with little concrete detail
        0.15 - 0.30  : Moderate — reasonably specific
        > 0.30       : High     — concrete, entity-rich requirement statements
    """
    if requirements is None:
        return {'mean': None, 'std': None, 'requirements_evaluated': 0,
                'per_req': [],
                'interpretation': 'requirements.json not found — run extraction.py first',
                'skipped': True}

    req_list = requirements.get('requirements', [])
    if not req_list:
        return {'mean': 0.0, 'std': 0.0, 'requirements_evaluated': 0,
                'per_req': [], 'interpretation': 'No requirements extracted',
                'skipped': False}

    densities = []
    for req in req_list:
        doc         = nlp(req.get('text', ''))
        token_count = len([t for t in doc if not t.is_space])
        np_count    = len(list(doc.noun_chunks))
        densities.append(round(np_count / token_count, 4) if token_count else 0.0)

    mean = round(sum(densities) / len(densities), 4)
    std  = round((sum((d - mean) ** 2 for d in densities) / len(densities)) ** 0.5, 4)

    if mean >= 0.30:   interp = "High — concrete, entity-rich requirement statements"
    elif mean >= 0.15: interp = "Moderate — reasonably specific"
    else:              interp = "Low — vague requirements with little concrete detail"

    return {'mean': mean, 'std': std, 'requirements_evaluated': len(densities),
            'per_req': densities, 'interpretation': interp, 'skipped': False}


# Orchestrator
def evaluate_run(run_dir: Path) -> dict:
    """
    Run all five metrics against a single SimAgents output directory.

    Loads ML models once, computes all metrics, saves evaluation.json,
    and prints a terminal summary.

    Args:
        run_dir (Path): Path to a completed run output directory.

    Returns:
        dict: Full evaluation results ready for JSON serialisation.
    """
    print(f"\nEvaluating: {run_dir.name}")

    run_data     = load_run(run_dir)
    dialogue     = run_data['dialogue']
    requirements = run_data['requirements']
    run_id       = run_data['run_id']
    scenario     = _load_scenario(run_id)

    sbert_model, sbert_util = _load_sbert()
    nlp = _load_spacy()

    re_turns, user_turns = _split_messages(dialogue)

    coherence   = compute_coherence(dialogue, sbert_model, sbert_util)
    re_info     = compute_informativeness(re_turns, nlp)
    user_info   = compute_informativeness(user_turns, nlp)

    if scenario:
        coverage = compute_coverage(dialogue, scenario, sbert_model, sbert_util)
    else:
        coverage = {
            'score': None, 'covered': [], 'not_covered': [],
            'total_expected': 0, 'by_type': {},
            'similarity_threshold': 0.4,
            'interpretation': 'Scenario not found — coverage skipped'
        }

    specificity = compute_specificity(requirements, nlp)

    results = {
        'run_id':         run_id,
        'evaluated_at':   datetime.now().isoformat(),
        'dialogue_turns': len(re_turns) + len(user_turns),
        'metrics': {
            'coherence':               coherence,
            're_informativeness':      re_info,
            'user_informativeness':    user_info,
            'requirement_coverage':    coverage,
            'requirement_specificity': specificity
        },
        'summary': {
            'coherence_mean':                coherence['mean'],
            're_informativeness_mean':       re_info['mean'],
            'user_informativeness_mean':     user_info['mean'],
            'requirement_coverage':          coverage.get('score'),
            'requirement_specificity_mean':  specificity.get('mean')
        }
    }

    output_file = run_dir / "evaluation.json"
    with open(output_file, 'w', encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    _print_summary(results)
    print(f"\n  Saved to: {output_file}")

    return results


def _print_summary(results: dict):
    """Print a formatted summary of metric scores to the terminal."""
    s = results['summary']
    cov_detail = results['metrics']['requirement_coverage'].get('by_type', {})

    TYPE_DISPLAY = {
        'FunctionalRequirement':    'Functional',
        'NonFunctionalRequirement': 'Non-Functional',
        'Constraint':               'Constraint',
    }

    print(f"\n{'─'*60}")
    print(f"  RESULTS: {results['run_id']}")
    print(f"{'─'*60}")
    print(f"  Dialogue turns          : {results['dialogue_turns']}")
    print(f"{'─'*60}")
    print(f"  Coherence               : {s['coherence_mean']}")
    print(f"  RE Informativeness      : {s['re_informativeness_mean']}")
    print(f"  User Informativeness    : {s['user_informativeness_mean']}")
    cov = s['requirement_coverage']
    print(f"  Requirement Coverage    : {cov if cov is not None else 'N/A'}")
    for rtype, score in cov_detail.items():
        label = TYPE_DISPLAY.get(rtype, rtype)
        print(f"    {label:<22}: {score}")
    spec = s['requirement_specificity_mean']
    print(f"  Requirement Specificity : {spec if spec is not None else 'N/A (run extraction first)'}")
    print(f"{'─'*60}")


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SimAgents Evaluation — computes dialogue and requirement quality metrics for a single run"
    )
    parser.add_argument('run_dir', help="Path to a completed run output directory")
    args = parser.parse_args()

    target = Path(args.run_dir)
    if not target.is_dir():
        print(f"[ERROR] {target} is not a directory.")
        sys.exit(1)

    evaluate_run(target)

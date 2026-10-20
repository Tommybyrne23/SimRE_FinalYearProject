"""
server.py - SimAgents Web Server

Flask backend that wraps the SimAgents pipeline and exposes it as a REST API
for the browser-based dashboard. Supports real-time dialogue streaming via
Server-Sent Events (SSE).

Endpoints:
    GET  /api/config            — returns personas.json + scenarios.json merged
    POST /api/run               — starts a simulation, streams dialogue via SSE
    GET  /api/results           — lists all completed runs from outputs/
    GET  /api/results/<run_id>  — returns dialogue + requirements + evaluation for one run
    DELETE /api/results/<run_id> — deletes a run directory and all its files
    POST /api/evaluate/<run_id> — triggers automatic evaluation for a completed run
    POST /api/manual-eval/<run_id> — saves manual Likert-scale evaluation ratings
    GET  /api/compare           — returns comparison.json if it exists

Usage:
    pip install flask flask-cors
    python server.py
    Then open http://localhost:5000 in your browser.
"""

import asyncio
import json
import os
import sys
import queue
import threading
from pathlib import Path
from datetime import datetime

from flask import Flask, jsonify, request, Response, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents import create_model_client, create_re_agent, create_user_agent
from session import SimAgentsSession
from extraction import extract_requirements_llm

# Try to import evaluation — gracefully degrade if deps missing
try:
    from evaluation import evaluate_run
    EVAL_AVAILABLE = True
except Exception:
    EVAL_AVAILABLE = False

app = Flask(__name__, static_folder="ui", static_url_path="")
CORS(app)

CONFIG_DIR  = Path("config")
OUTPUTS_DIR = Path("outputs")


# ── Config ─────────────────────────────────────────────────────────────────

@app.route("/api/config")
def get_config():
    """Return personas and scenarios for the frontend dropdowns."""
    try:
        with open(CONFIG_DIR / "personas.json", encoding="utf-8") as f:
            personas = json.load(f)
        with open(CONFIG_DIR / "scenarios.json", encoding="utf-8") as f:
            scenarios = json.load(f)
        return jsonify({"personas": personas, "scenarios": scenarios})
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500


# ── Run simulation with SSE streaming ──────────────────────────────────────

@app.route("/api/run", methods=["POST"])
def run_simulation():
    """
    Start a simulation and stream dialogue turns as Server-Sent Events.

    Request body (JSON):
        scenario_index  (int): index into scenarios array
        re_persona_index (int): index into re_personas array
        user_persona_index (int): index into user_personas array

    SSE event types:
        data: {"type": "message", "source": "...", "content": "..."}
        data: {"type": "requirements", "requirements": [...]}
        data: {"type": "done", "run_id": "..."}
        data: {"type": "error", "message": "..."}
    """
    body = request.get_json()
    scenario_idx    = int(body.get("scenario_index", 0))
    re_idx          = int(body.get("re_persona_index", 0))
    user_idx        = int(body.get("user_persona_index", 0))

    # Load config
    with open(CONFIG_DIR / "personas.json", encoding="utf-8") as f:
        personas_data = json.load(f)
    with open(CONFIG_DIR / "scenarios.json", encoding="utf-8") as f:
        scenario_data = json.load(f)

    scenario    = scenario_data["scenarios"][scenario_idx]
    re_persona  = personas_data["re_personas"][re_idx]
    user_persona = personas_data["user_personas"][user_idx]

    run_id = f"run_{scenario['id']}_{re_persona['id']}_vs_{user_persona['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # SSE queue — background thread posts events here
    event_q = queue.Queue()

    def run_async():
        """Run the async AutoGen session in a background thread."""
        async def _run():
            use_local    = os.getenv("USE_LOCAL", "true").lower() == "true"
            api_key      = os.getenv("OPENAI_API_KEY")
            local_model  = os.getenv("LOCAL_MODEL", "llama3.2")
            openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            temperature  = float(os.getenv("TEMPERATURE", "0.7"))
            seed_env     = os.getenv("SEED")
            seed         = int(seed_env) if seed_env else None
            model_name   = local_model if use_local else openai_model

            re_model   = create_model_client(use_local, api_key, model_name, temperature, seed)
            user_model = create_model_client(use_local, api_key, model_name, temperature, seed)

            re_agent   = create_re_agent(re_model, scenario, re_persona)
            user_agent = create_user_agent(user_model, user_persona, scenario)

            # Monkey-patch session to intercept messages as they stream
            from autogen_agentchat.teams import RoundRobinGroupChat
            from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
            from autogen_agentchat.messages import TextMessage
            from autogen_agentchat.base import TaskResult

            OUTPUTS_DIR.mkdir(exist_ok=True)
            out_dir = OUTPUTS_DIR / run_id
            out_dir.mkdir(parents=True, exist_ok=True)

            participants = [re_agent, user_agent]
            max_turns    = scenario["constraints"]["max_turns"]
            termination  = (
                MaxMessageTermination(max_messages=max_turns) |
                TextMentionTermination("REQUIREMENTS_COMPLETE")
            )
            team = RoundRobinGroupChat(participants=participants, termination_condition=termination)
            task = TextMessage(content="Begin the requirements elicitation interview.", source="user")

            # Send model name to frontend before dialogue starts
            event_q.put(json.dumps({"type": "meta", "model": model_name}))

            saved_messages = []

            async for message in team.run_stream(task=task):
                if isinstance(message, TaskResult):
                    # Save dialogue
                    dialogue_data = {
                        "run_id": run_id,
                        "scenario": scenario["name"],
                        "model": model_name,
                        "timestamp": datetime.now().isoformat(),
                        "total_messages": len(message.messages),
                        "messages": [
                            {
                                "source": m.source if hasattr(m, "source") else "unknown",
                                "content": m.content if hasattr(m, "content") else str(m)
                            }
                            for m in message.messages
                            if not (hasattr(m, "source") and m.source == "user")
                        ]
                    }
                    with open(out_dir / "dialogue.json", "w", encoding="utf-8") as f:
                        json.dump(dialogue_data, f, indent=2, ensure_ascii=False)

                    # Extract requirements
                    try:
                        reqs = await extract_requirements_llm(out_dir / "dialogue.json")
                        event_q.put(json.dumps({"type": "requirements", "requirements": reqs}))
                    except Exception as e:
                        event_q.put(json.dumps({"type": "error", "message": f"Extraction failed: {e}"}))

                    event_q.put(json.dumps({"type": "done", "run_id": run_id}))
                    event_q.put(None)  # sentinel
                    return

                else:
                    if hasattr(message, "source") and message.source != "user":
                        content = message.content or ""
                        # Strip the REQUIREMENTS_COMPLETE sentinel from display
                        display = content.replace("REQUIREMENTS_COMPLETE", "").strip()
                        if display:
                            event_q.put(json.dumps({
                                "type": "message",
                                "source": message.source,
                                "content": display
                            }))

        try:
            asyncio.run(_run())
        except Exception as e:
            event_q.put(json.dumps({"type": "error", "message": str(e)}))
            event_q.put(None)

    thread = threading.Thread(target=run_async, daemon=True)
    thread.start()

    def generate():
        while True:
            item = event_q.get()
            if item is None:
                break
            yield f"data: {item}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Results ────────────────────────────────────────────────────────────────

@app.route("/api/results")
def list_results():
    """List all completed runs that have a dialogue.json."""
    OUTPUTS_DIR.mkdir(exist_ok=True)
    runs = []
    for run_dir in sorted(OUTPUTS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        dialogue_file = run_dir / "dialogue.json"
        if not dialogue_file.exists():
            continue
        try:
            with open(dialogue_file, encoding="utf-8") as f:
                d = json.load(f)
            run_info = {
                "run_id":    d.get("run_id", run_dir.name),
                "scenario":  d.get("scenario", "Unknown"),
                "model":     d.get("model", "unknown"),
                "timestamp": d.get("timestamp", ""),
                "total_messages": d.get("total_messages", 0),
                "has_requirements": (run_dir / "requirements.json").exists(),
                "has_evaluation":   (run_dir / "evaluation.json").exists(),
                "has_manual_eval":  (run_dir / "manual_evaluation.json").exists(),
            }
            # Pull evaluation summary if available
            eval_file = run_dir / "evaluation.json"
            if eval_file.exists():
                with open(eval_file, encoding="utf-8") as f:
                    ev = json.load(f)
                run_info["metrics"] = ev.get("summary", {})
            runs.append(run_info)
        except Exception:
            continue
    return jsonify(runs)


@app.route("/api/results/<run_id>")
def get_result(run_id):
    """Return full dialogue, requirements, and evaluation for a single run."""
    run_dir = OUTPUTS_DIR / run_id
    if not run_dir.exists():
        return jsonify({"error": "Run not found"}), 404

    result = {}

    dialogue_file = run_dir / "dialogue.json"
    if dialogue_file.exists():
        with open(dialogue_file, encoding="utf-8") as f:
            result["dialogue"] = json.load(f)

    req_file = run_dir / "requirements.json"
    if req_file.exists():
        with open(req_file, encoding="utf-8") as f:
            result["requirements"] = json.load(f)

    eval_file = run_dir / "evaluation.json"
    if eval_file.exists():
        with open(eval_file, encoding="utf-8") as f:
            result["evaluation"] = json.load(f)

    manual_file = run_dir / "manual_evaluation.json"
    if manual_file.exists():
        with open(manual_file, encoding="utf-8") as f:
            result["manual_evaluation"] = json.load(f)

    return jsonify(result)


@app.route("/api/manual-eval/<run_id>", methods=["POST"])
def save_manual_eval(run_id):
    """Save manual evaluation ratings for a completed run."""
    run_dir = OUTPUTS_DIR / run_id
    if not run_dir.exists():
        return jsonify({"error": "Run not found"}), 404

    body = request.get_json()
    ratings = body.get("ratings", {})

    manual_data = {
        "run_id": run_id,
        "evaluated_at": datetime.now().isoformat(),
        "evaluator": body.get("evaluator", "anonymous"),
        "ratings": ratings,
    }

    manual_file = run_dir / "manual_evaluation.json"
    with open(manual_file, "w", encoding="utf-8") as f:
        json.dump(manual_data, f, indent=2, ensure_ascii=False)

    return jsonify(manual_data)


@app.route("/api/evaluate/<run_id>", methods=["POST"])
def evaluate(run_id):
    """Trigger evaluation for a completed run."""
    if not EVAL_AVAILABLE:
        return jsonify({"error": "Evaluation dependencies not installed. Run: pip install sentence-transformers spacy && python -m spacy download en_core_web_sm"}), 400

    run_dir = OUTPUTS_DIR / run_id
    if not run_dir.exists():
        return jsonify({"error": "Run not found"}), 404

    try:
        result = evaluate_run(run_dir)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/compare")
def get_compare():
    """Return comparison.json from the outputs directory, enriched with model info."""
    compare_file = OUTPUTS_DIR / "comparison.json"
    if not compare_file.exists():
        compare_file = Path("comparison.json")
    if compare_file.exists():
        with open(compare_file, encoding="utf-8") as f:
            data = json.load(f)
        # Enrich with model info from dialogue.json
        models = {}
        for run_id in data.get("runs", []):
            dlg_file = OUTPUTS_DIR / run_id / "dialogue.json"
            if dlg_file.exists():
                try:
                    with open(dlg_file, encoding="utf-8") as f:
                        d = json.load(f)
                    models[run_id] = d.get("model", "unknown")
                except Exception:
                    models[run_id] = "unknown"
        data["models"] = models
        return jsonify(data)
    return jsonify({"error": "No comparison data found. Run comparison.py first."}), 404


@app.route("/api/results/<run_id>", methods=["DELETE"])
def delete_result(run_id):
    """Delete a completed run directory and all its files."""
    run_dir = OUTPUTS_DIR / run_id
    if not run_dir.exists():
        return jsonify({"error": "Run not found"}), 404
    try:
        import shutil
        shutil.rmtree(run_dir)
        return jsonify({"status": "deleted", "run_id": run_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Serve frontend ─────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("ui", "index.html")


if __name__ == "__main__":
    print("\nSimAgents Dashboard")
    print("  http://localhost:5000\n")
    app.run(debug=True, port=5000, threaded=True)

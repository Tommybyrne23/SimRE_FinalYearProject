"""
extraction.py - LLM-Based Requirements Extraction

Performs a direct LLM call to extract and classify requirements from a completed dialogue transcript. Operates independently of the AutoGen
AgentChat pipeline, reading the saved dialogue.json and writing requirements.json to the same run output directory.

Classification uses the three requirement types defined in scenarios.json:
    FunctionalRequirement     — what the system must do
    NonFunctionalRequirement  — how well the system must do it
    Constraint                — external rules, regulations, or integration limits
    the system must comply with
"""

import json
from pathlib import Path
from dotenv import load_dotenv
import os

from autogen_core.models import UserMessage
from agents import create_model_client


async def extract_requirements_llm(dialogue_file: Path, model_client=None):
    """
    Extract and classify requirements from a dialogue transcript using a direct LLM call.

    Reads the dialogue.json produced by session.py, constructs a structured
    extraction prompt, and writes the results to requirements.json in the
    same output directory.

    Args:
        dialogue_file (Path): Path to the dialogue.json file for this run.
        model_client (optional): Pre-created LLM client. If None, creates a new
            one from environment variables (backwards compatible but less efficient
            when called repeatedly in a batch).

    Returns:
        list: Extracted requirements as a list of dicts, each containing:
            - 'id': Sequential identifier (e.g., REQ_001)
            - 'text': Requirement statement
            - 'type': One of FunctionalRequirement, NonFunctionalRequirement, Constraint
            - 'source': Agent name that raised the requirement
    """
    load_dotenv()

    use_local  = os.getenv("USE_LOCAL", "true").lower() == "true"
    api_key    = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("LOCAL_MODEL", "llama3.2") if use_local else os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if model_client is None:
        model_client = create_model_client(use_local, api_key, model_name)

    with open(dialogue_file, 'r', encoding="utf-8") as f:
        data = json.load(f)

    messages = data['messages']

    # Format dialogue as a clean turn-by-turn transcript for the prompt
    dialogue_text = "\n".join([
        f"{msg.get('source', '?')}: {msg.get('content', '')}"
        for msg in messages
    ])

    prompt = f"""Extract ALL requirements from this requirements elicitation dialogue.

DIALOGUE:
{dialogue_text}

CLASSIFICATION RULES:
- FunctionalRequirement  — something the system must do (features, behaviours, actions)
- NonFunctionalRequirement — how well the system must do it (performance, usability, security, availability)
- Constraint — an external rule, regulation, integration limit, or policy the system must comply with

EXTRACTION RULES:
- Find both explicit requirements ("the system must/should/shall") and implicit ones (clearly needed but not directly stated)
- Record which agent raised each requirement in the source field
- Each requirement must be distinct — if the same behaviour is mentioned multiple times across turns, extract it once only
- Do not extract requirements from summaries or recaps if they were already captured from earlier in the dialogue

OUTPUT ONLY a JSON array with no explanation and no markdown fences:
[
  {{"id": "REQ_001", "text": "Send SMS and email reminders before appointments", "type": "FunctionalRequirement", "source": "User_cooperative_expert"}},
  {{"id": "REQ_002", "text": "System must comply with GDPR for customer data storage", "type": "Constraint", "source": "User_cooperative_expert"}},
  {{"id": "REQ_003", "text": "Booking interface must be usable on mobile devices", "type": "NonFunctionalRequirement", "source": "RE_experienced_methodical"}}
]"""

    try:
        response = await model_client.create(
            messages=[UserMessage(content=prompt, source="user")]
        )
        llm_output = response.content

    except Exception as e:
        print(f"Extraction error: {e}")
        llm_output = "[]"

    # Parse JSON safely — strip markdown fences if the model adds them despite instructions
    try:
        cleaned = llm_output.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        requirements = json.loads(cleaned)
        if isinstance(requirements, dict):
            requirements = requirements.get('requirements', [])
        if not isinstance(requirements, list):
            requirements = []
    except Exception:
        requirements = []

    output_file = dialogue_file.parent / "requirements.json"
    output_data = {
        "run_id": data['run_id'],
        "extraction_method": "LLM-direct",
        "model": model_name,
        "total_requirements": len(requirements),
        "requirements": requirements
    }

    with open(output_file, 'w', encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    return requirements

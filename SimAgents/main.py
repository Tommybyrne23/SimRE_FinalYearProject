"""
SimAgents Main Entry Point
Follows async patterns from: 1.2-Asynchronous-Run.py & AgentChat_First_Agent.ipynb
"""

import asyncio
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Import our modules
import sys
sys.path.append('src')
from agents import create_model_client, create_re_agent, create_user_agent
from session import SimAgentsSession
from extraction import extract_requirements_llm


# ── Experiment Configuration ──────────────────────────────────────────────────
# Set these indices to select the combination you want to run.
#
# SCENARIO_INDEX:
#   0 = barber_booking
#   1 = clinic_patient_portal
#   2 = library_system
#   3 = student_portal_enhancement
#   4 = banking_app_modernisation
#
# RE_PERSONA_INDEX:
#   0 = experienced_methodical (David - Senior)
#   1 = adaptive_generalist    (Eric  - Mid-level)
#   2 = junior_inexperienced   (Rachel - Junior)
#
# USER_PERSONA_INDEX (must match scenario):
#   Barbershop : 0, 1, 2
#   Clinic     : 3, 4, 5
#   Library    : 6, 7, 8
#   Portal     : 9, 10, 11
#   Banking    : 12, 13, 14
#   Within each group: first = Cooperative Expert, second = Vague Novice, third = Resistant Expert

SCENARIO_INDEX     = 0
RE_PERSONA_INDEX   = 2
USER_PERSONA_INDEX = 2
# ─────────────────────────────────────────────────────────────────────────────


async def main():
    """Main execution function for SimAgentsSession."""

    # Load environment
    load_dotenv()

    # LLM configuration
    use_local    = os.getenv("USE_LOCAL", "true").lower() == "true"
    api_key      = os.getenv("OPENAI_API_KEY")
    local_model  = os.getenv("LOCAL_MODEL", "llama3.2")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature  = float(os.getenv("TEMPERATURE", "0.7"))
    seed         = int(os.getenv("SEED")) if os.getenv("SEED") else None
    model_name   = local_model if use_local else openai_model

    # Load configurations
    with open("config/personas.json", 'r', encoding="utf-8") as f:
        personas_data = json.load(f)

    with open("config/scenarios.json", 'r', encoding="utf-8") as f:
        scenario_data = json.load(f)

    # Select scenario and personas by index
    scenario     = scenario_data['scenarios'][SCENARIO_INDEX]
    re_persona   = personas_data['re_personas'][RE_PERSONA_INDEX]
    user_persona = personas_data['user_personas'][USER_PERSONA_INDEX]

    # Build run ID from persona combination
    run_id = f"run_{scenario['id']}_{re_persona['id']}_vs_{user_persona['id']}"
    if seed is not None:
        run_id += f"_seed{seed}"

    print("\nSimAgents - Requirements Elicitation Simulation")
    print(f"Model:        {'LOCAL - ' + model_name if use_local else 'OpenAI - ' + model_name}")
    print(f"Scenario:     {scenario['name']}")
    print(f"RE Persona:   {re_persona['name']}")
    print(f"User Persona: {user_persona['name']}")
    print(f"Run ID:       {run_id}\n")

    # Create model clients
    re_model   = create_model_client(use_local, api_key, model_name, temperature, seed)
    user_model = create_model_client(use_local, api_key, model_name, temperature, seed)

    # Create agents
    re_agent = create_re_agent(re_model, scenario, re_persona)
    user     = create_user_agent(user_model, user_persona, scenario)

    # Create and run session
    session = SimAgentsSession(
        re_agent=re_agent,
        user_agents=[user],
        scenario=scenario,
        run_id=run_id,
        model_name=model_name
    )

    result = await session.run_session()

    # Extract requirements
    dialogue_file = session.output_dir / "dialogue.json"
    if dialogue_file.exists():
        requirements = await extract_requirements_llm(dialogue_file)
        print(f"Extracting requirements...  {len(requirements)} extracted.\n")


if __name__ == "__main__":
    # Run async main (pattern from notebooks)
    asyncio.run(main())

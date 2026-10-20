"""
agents.py - Agent Creation Module for SimAgents Project

This module provides factory functions to create AutoGen agents for requirements
elicitation simulation. It handles both Requirements Engineer (RE) agents and
User/Stakeholder agents with configurable personas.

Architecture Pattern:
- Follows AutoGen 0.4 AgentChat API patterns
- Uses AssistantAgent as the base agent type
- Supports both local (Ollama) and cloud (OpenAI) LLM backends
- Pattern source: AgentChat_First_Agent.ipynb & Agents-in-Autogen.ipynb

Persona Design:
- RE personas define analyst experience level, questioning style, and tone
- User personas define stakeholder goals, domain knowledge, and cooperation level
- Both persona types are loaded from config/personas.json and injected at agent creation
- This separation allows systematic comparison of RE strategies against diverse stakeholders
  as recommended by Dabrowski et al. (2025) and Karolita et al. (2024)
"""


from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.ollama import OllamaChatCompletionClient


def create_model_client(use_local: bool, api_key: str = None, model_name: str = None,
                        temperature: float = 0.7, seed: int = None):
    """
    Factory function to create LLM client based on configuration.

    LLM Configuration Rationale:
        temperature: Controls response randomness (0.0 = deterministic, 1.0 = highly varied).
            Default 0.7 balances natural-sounding dialogue with consistency.
            For reproducibility experiments, set to 0.0 with a fixed seed.
        seed: When provided alongside temperature=0, ensures identical outputs across
            repeated runs of the same configuration, critical for controlled evaluation
            as described in the project specification.

    Args:
        use_local (bool): If True, creates Ollama client; if False, creates OpenAI client.
        api_key (str): OpenAI API key, required if use_local is False.
        model_name (str): Name of the model to use.
        temperature (float): Sampling temperature for LLM responses. Default 0.7.
        seed (int, optional): Random seed for reproducible outputs. If None, outputs vary.

    Returns:
        OllamaChatCompletionClient or OpenAIChatCompletionClient:
            Configured model client ready for agent creation.
    """

    if use_local:
        # Ollama local model, runs on localhost:11434
        # Assumes Ollama is already running in the background
        # Temperature and seed passed through to control output variance
        return OllamaChatCompletionClient(
            model=model_name or "llama3.2",
            base_url="http://localhost:11434",
            temperature=temperature,
            seed=seed
        )

    else:
        # OpenAI API — seed supported natively for deterministic outputs
        return OpenAIChatCompletionClient(
            model=model_name or "gpt-4o-mini",
            api_key=api_key,
            temperature=temperature,
            seed=seed
        )


def create_re_agent(model_client, scenario: dict, re_persona: dict = None):
    """
    Creates a Requirements Engineer (RE) agent configured to conduct elicitation interviews.

    The RE agent's behaviour is driven by two inputs:
        1. The scenario — defines what project is being discussed.
        2. The RE persona — defines how the agent conducts the interview (experience level,
           questioning style, tone). This enables systematic comparison of analyst strategies,
           directly addressing RQ2.

    If no RE persona is provided, the agent defaults to a neutral professional style,
    preserving backwards compatibility with existing code.

    Args:
        model_client: LLM client created by create_model_client().
        scenario (dict): Scenario configuration containing:
            - 'name': Project name
            - 'context': Background information
            - 'goal': Overall objective
            - 'constraints': Dictionary with 'min_requirements' and 'max_turns'
        re_persona (dict, optional): RE persona configuration containing:
            - 'id': Unique identifier (e.g., 'experienced_methodical')
            - 'name': Display name
            - 'experience_level': 'senior', 'mid', or 'junior'
            - 'tone': Communication tone (e.g., 'formal, patient, precise')
            - 'questioning_style': 'structured', 'adaptive', or 'surface-level'
            - 'domain_knowledge': 'high', 'medium', or 'low'
            - 'behavior': Dict with probing_depth, summarisation_frequency,
                          clarification_approach, adapts_to_stakeholder

    Returns:
        AssistantAgent: Configured RE agent ready to conduct interviews.
    """

    # Build persona-specific instructions if a persona is provided.
    # These instructions modify how the agent interviews, not what it asks about.
    # This separation of scenario (what) and persona (how) is key to the framework design.
    if re_persona:
        behavior = re_persona.get('behavior', {})

        # Translate structured persona config into natural language LLM instructions
        persona_instructions = f"""
YOUR ANALYST PROFILE:
- Experience Level: {re_persona.get('experience_level', 'mid')}
- Questioning Style: {re_persona.get('questioning_style', 'adaptive')}
- Tone: {re_persona.get('tone', 'professional')}
- Domain Knowledge: {re_persona.get('domain_knowledge', 'medium')}

HOW YOU CONDUCT INTERVIEWS:
- Probing Depth: {behavior.get('probing_depth', 'medium')} — {"push deeper when answers are vague; ask 'why', 'how', 'what specifically'" if behavior.get('probing_depth') == 'deep' else "accept reasonable answers but follow up when something is clearly missing" if behavior.get('probing_depth') == 'medium' else "accept answers at face value; do not probe deeper"}
- Summarisation: {"Frequently summarise what you have heard to confirm understanding" if behavior.get('summarisation_frequency') == 'high' else "Occasionally summarise key points" if behavior.get('summarisation_frequency') == 'medium' else "Rarely summarise; move on to the next question"}
- Clarification: {behavior.get('clarification_approach', 'ask follow-up questions when unclear')}
- Stakeholder Adaptation: {"Adjust your questioning style based on how the stakeholder responds — simpler language for novices, more technical for experts" if behavior.get('adapts_to_stakeholder') else "Maintain your standard interview style regardless of how the stakeholder responds"}
"""
        agent_name = f"RE_{re_persona['id']}"
        agent_description = f"Requirements Engineer ({re_persona.get('experience_level', 'mid')}-level): {re_persona.get('questioning_style', 'adaptive')} style"
    else:
        # Default neutral RE behaviour — no persona applied
        persona_instructions = """
YOUR ANALYST PROFILE:
- Experience Level: mid
- Questioning Style: adaptive
- Tone: professional
- Conduct a thorough but flexible elicitation interview.
"""
        agent_name = "RE_Analyst"
        agent_description = "Requirements Engineer conducting elicitation"

    # System message combines scenario context (what to ask about)
    # with persona instructions (how to ask), keeping concerns cleanly separated.
    # KEY AREAS TO EXPLORE is intentionally omitted — the agent explores based on
    # scenario context and persona alone, not a provided checklist. This ensures
    # that FR/NFR/Constraint coverage differences in evaluation reflect genuine
    # persona behaviour rather than task framing.
    system_message = f"""You are a Requirements Engineer conducting a requirements elicitation interview.

PROJECT: {scenario['name']}
CONTEXT: {scenario['context']}
GOAL: {scenario['goal']}
{persona_instructions}
INTERVIEW STRATEGY:
1. Start with open-ended questions about high-level goals
2. Ask follow-up questions to clarify vague responses
3. Probe for specific details with "What", "How", "Why" questions
4. Ask for concrete examples: "Can you give me an example?"
5. Identify constraints: "Are there any limitations or restrictions?"
6. Confirm understanding: "So if I understand correctly..."

RULES:
- Ask ONE question at a time
- Reference previous answers to show you are listening
- When you have gathered {scenario['constraints']['min_requirements']} clear requirements, say REQUIREMENTS_COMPLETE on its own line as your entire final message — do not add anything before or after it.

Keep the conversation natural and appropriate to your analyst profile above."""

    return AssistantAgent(
        name=agent_name,
        model_client=model_client,
        system_message=system_message,
        description=agent_description
    )


def _detect_archetype(persona_id: str) -> str:
    """
    Detect user archetype from persona ID suffix.
    All persona IDs end with one of three archetype identifiers,
    optionally prefixed with a scenario name (e.g., clinic_, library_).

    Args:
        persona_id (str): Persona identifier from personas.json.

    Returns:
        str: One of 'cooperative_expert', 'vague_novice', 'resistant_expert'.
             Defaults to 'cooperative_expert' if no match (safe fallback).
    """
    for archetype in ('cooperative_expert', 'vague_novice', 'resistant_expert'):
        if persona_id.endswith(archetype):
            return archetype
    return 'cooperative_expert'


def _format_ground_truth_for_stakeholder(ground_truth: list) -> str:
    """
    Format ground truth requirements grouped by type for injection into the
    user agent's system prompt. The agent receives these as knowledge they
    hold about their own needs but must express in natural stakeholder voice.

    Args:
        ground_truth (list): List of requirement dicts from scenarios.json,
            each with 'type' and 'statement' fields.

    Returns:
        str: Formatted multi-line string grouped by requirement type.
    """
    by_type = {
        'FunctionalRequirement':    [],
        'NonFunctionalRequirement': [],
        'Constraint':               []
    }
    for req in ground_truth:
        rtype = req.get('type', 'FunctionalRequirement')
        if rtype in by_type:
            by_type[rtype].append(req.get('statement', ''))

    sections = []
    if by_type['FunctionalRequirement']:
        sections.append("Things you want the system to do (functional needs):")
        sections.extend(f"  - {s}" for s in by_type['FunctionalRequirement'])
    if by_type['NonFunctionalRequirement']:
        sections.append("\nHow well it needs to work (quality / performance needs):")
        sections.extend(f"  - {s}" for s in by_type['NonFunctionalRequirement'])
    if by_type['Constraint']:
        sections.append("\nRules and limits you have to comply with:")
        sections.extend(f"  - {s}" for s in by_type['Constraint'])

    return "\n".join(sections)


# Archetype-specific disclosure behaviour layered on top of the four behaviour
# flags. Controls HOW knowledge is revealed during dialogue, not WHAT is known.
# All archetypes have full knowledge of the ground truth; only revelation differs.
ARCHETYPE_DISCLOSURE = {
    'cooperative_expert': (
        "You are eager to help and want this project to succeed. When asked an "
        "open question, voluntarily bring up multiple related needs that come "
        "to mind, even ones not directly asked about. Give specific examples "
        "and concrete details. If you remember a constraint or quality concern "
        "while answering, mention it. Your goal is for the interviewer to leave "
        "with a complete picture."
    ),
    'vague_novice': (
        "You know what you want but struggle to express it precisely. Keep "
        "answers short and general. Use everyday language — never technical "
        "terms. You will often forget to mention important needs unless the "
        "interviewer asks about them directly. If the interviewer accepts a "
        "vague answer without probing, do not volunteer the missing detail. "
        "Sometimes ask the interviewer to clarify what they mean before answering."
    ),
    'resistant_expert': (
        "You know exactly what you need but you are reluctant to share. Give "
        "minimal one or two sentence answers. Do NOT volunteer information "
        "beyond what was directly asked. Do NOT bring up constraints, quality "
        "concerns, or related needs unless the interviewer specifically asks "
        "about that topic. Answer accurately when pressed but offer no extras. "
        "You are not hostile — you simply do not elaborate."
    ),
}


def create_user_agent(model_client, persona: dict, scenario: dict):
    """
    Creates a User/Stakeholder agent with persona-driven behaviour AND
    scenario-grounded knowledge of the ground truth requirements.

    The user agent is given full knowledge of the scenario's ground truth
    requirements (so the RE agent has something real to elicit), but the
    archetype controls HOW that knowledge is revealed during dialogue:
      - cooperative_expert → volunteers proactively, articulates clearly
      - vague_novice       → expresses imprecisely, forgets unless probed
      - resistant_expert   → minimal answers, no volunteering

    This addresses the supervisor's 10 April feedback: "User agents must
    know the ground truth requirements." Without scenario-grounded
    knowledge, the user agent can only respond from the four high-level
    goal bullets in personas.json, which makes coverage evaluation
    structurally impossible.

    User agents simulate real stakeholders with varying characteristics:
    - Communication styles (clear vs vague, detailed vs brief)
    - Domain knowledge levels (expert vs novice)
    - Cooperation levels (proactive vs reactive vs resistant)
    - Behavioural traits (detail-oriented vs high-level)

    Three user personas are defined, each targeting a different elicitation challenge:
    - Cooperative Expert: baseline productive stakeholder (Karolita et al., 2024)
    - Vague Novice: tests extraction from unclear responses (Korn et al., 2025)
    - Resistant Expert: tests handling of low-cooperation, high-knowledge stakeholders
      (Oriol et al., 2025; Karolita et al., 2024)

    Args:
        model_client: LLM client created by create_model_client().
        persona (dict): Persona configuration from config/personas.json user_personas section.
            Required fields: id, name, role, background, communication_style,
            cooperation_level, domain_knowledge, goals, behavior.
            behavior dict requires: detail_oriented, provides_examples,
            mentions_constraints, asks_clarifying_questions.
        scenario (dict): Scenario configuration from config/scenarios.json
            containing 'ground_truth_requirements' list (each with 'type'
            and 'statement' fields).

    Returns:
        AssistantAgent: Configured user agent that acts according to persona
            and has knowledge of the scenario's ground truth requirements.

    Example:
        personas  = json.load(open('config/personas.json', encoding="utf-8"))
        scenarios = json.load(open('config/scenarios.json', encoding="utf-8"))
        persona  = next(p for p in personas['user_personas'] if p['id'] == 'resistant_expert')
        scenario = next(s for s in scenarios['scenarios'] if s['id'] == 'barber_booking')
        user = create_user_agent(model_client, persona, scenario)
    """

    # Build behavioural traits list based on persona configuration.
    # Translated to natural language instructions so the LLM understands the expected behaviour.
    behavior_traits = []

    # Trait 1: Level of detail provided in responses
    if persona['behavior']['detail_oriented']:
        behavior_traits.append("Provide specific details, numbers, and concrete examples when answering")
    else:
        behavior_traits.append("Give brief, vague answers — avoid specifics unless directly pressed")

    # Trait 2: Proactivity in volunteering additional information
    if persona['behavior']['provides_examples']:
        behavior_traits.append("Proactively mention related needs or problems without being asked")
    else:
        behavior_traits.append("Only answer what is directly asked — do not volunteer extra information")

    # Trait 3: Whether the stakeholder mentions constraints unprompted
    if persona['behavior']['mentions_constraints']:
        behavior_traits.append("Mention limitations, budget concerns, or technical constraints when relevant")
    else:
        behavior_traits.append("Do not mention constraints unless specifically asked about them")

    # Trait 4: Tendency to seek clarification before answering
    if persona['behavior']['asks_clarifying_questions']:
        behavior_traits.append("Ask for clarification when a question is unclear before answering")
    else:
        behavior_traits.append("Interpret questions as best you can and answer without asking for clarification")

    # Detect archetype from persona ID and look up disclosure behaviour.
    # Inject the scenario's ground truth as the stakeholder's domain knowledge.
    archetype               = _detect_archetype(persona['id'])
    knowledge_block         = _format_ground_truth_for_stakeholder(
        scenario.get('ground_truth_requirements', [])
    )
    disclosure_instructions = ARCHETYPE_DISCLOSURE[archetype]

    system_message = f"""You are {persona['name']}, a {persona['role']}.

BACKGROUND:
{persona['background']}

WHY YOU ARE HERE (your motivation for this meeting):
{chr(10).join(f"- {g}" for g in persona.get('goals', []))}

WHAT YOU KNOW ABOUT YOUR NEEDS:
You have lived experience with this problem. The list below is what you
actually need from the system — but it is written in formal language, NOT
the way you would say it. During the interview you must express these
needs in your own everyday words. NEVER use the words "shall", "must",
"the system", or quote any item below verbatim. Translate them into how
a real {persona['role']} would speak.

{knowledge_block}

COMMUNICATION STYLE: {persona['communication_style']}
DOMAIN KNOWLEDGE: {persona['domain_knowledge']}
COOPERATION LEVEL: {persona['cooperation_level']}

HOW YOU BEHAVE IN THIS INTERVIEW:
{chr(10).join(f"- {trait}" for trait in behavior_traits)}

HOW YOU DISCLOSE YOUR KNOWLEDGE:
{disclosure_instructions}

IMPORTANT:
- Stay in character throughout the entire conversation
- Do NOT break character or explain your persona to the interviewer
- Do NOT list multiple requirements in formal language
- Respond naturally as this person would in a real stakeholder meeting
- If asked about something not in your knowledge above, say you haven't
  thought about that — do not invent details
"""

    return AssistantAgent(
        name=f"User_{persona['id']}",
        model_client=model_client,
        system_message=system_message,
        description=f"Stakeholder: {persona['role']} (archetype: {archetype}, knowledge: {persona['domain_knowledge']})"
    )

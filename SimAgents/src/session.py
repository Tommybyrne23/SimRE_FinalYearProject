"""
session.py - Team Orchestration Module for SimAgents Project

This module implements the multi-agent team coordination using AutoGen's
RoundRobinGroupChat pattern. It manages the conversation flow between the
Requirements Engineer and User agents, handles termination conditions,
and persists dialogue data for analysis.

Architecture Pattern:
- Uses RoundRobinGroupChat for sequential turn-taking
- Implements termination conditions (max turns + keyword detection)
- Follows async/await patterns for non-blocking execution
- Pattern source: 1.-Understanding-MultiAgent-Systems.ipynb
"""

from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.base import TaskResult
from datetime import datetime
from pathlib import Path
import json


class SimAgentsSession:
    """
    Orchestrates Requirements Engineer and User agents in a structured dialogue.

    This class manages the complete lifecycle of a requirements elicitation session:
    1. Initialises agents and team configuration
    2. Creates output directories for data persistence
    3. Executes the dialogue using RoundRobinGroupChat
    4. Captures and saves conversation transcripts

    The RoundRobinGroupChat ensures agents take turns speaking in a predictable order,
    mimicking real interview dynamics where the interviewer asks, the stakeholder responds,
    and this repeats until a termination condition is met.

    Attributes:
        re_agent: The Requirements Engineer agent (interviewer)
        user_agents (list): List of User/Stakeholder agents (interviewees)
        scenario (dict): Scenario configuration with context and constraints
        run_id (str): Unique identifier for this session
        output_dir (Path): Directory where dialogue and requirements are saved
        team (RoundRobinGroupChat): The configured multi-agent team

    Pattern Reference:
        1.-Understanding-MultiAgent-Systems.ipynb - Shows RoundRobinGroupChat usage
        2.-SelectorGroupChat.ipynb - Alternative pattern (not used here)

    Example:
        team = SimAgentsSession(
            re_agent=re_agent,
            user_agents=[user_agent],
            scenario=scenario_config,
            run_id="experiment_001"
        )
        result = await team.run_session()
    """

    def __init__(self, re_agent, user_agents: list, scenario: dict, run_id: str = None, model_name: str = None):
        """
        Initialise the SimAgents team with agents and configuration.

        Args:
            re_agent: Requirements Engineer agent (created by create_re_agent)
            user_agents (list): List of user agents (created by create_user_agent)
            scenario (dict): Scenario configuration containing:
                - 'name': Scenario name for display
                - 'constraints': Dict with 'max_turns' for termination
            run_id (str, optional): Custom run identifier. If None, generates
                timestamp-based ID (e.g., "run_20260210_200000")
            model_name (str, optional): Name of the LLM model used for this run.
                Stored in dialogue.json for traceability.

        Side Effects:
            - Creates output directory at outputs/{run_id}/
            - Initialises RoundRobinGroupChat team

        Design Decision:
            Directory creation happens in __init__ to fail fast if filesystem
            permissions are incorrect, rather than waiting until dialogue completes.
        """
        # Store agent references for access throughout session
        self.re_agent = re_agent
        self.user_agents = user_agents
        self.scenario = scenario
        self.model_name = model_name

        # Generate unique run ID if not provided
        # Format: run_YYYYMMDD_HHMMSS (e.g., run_20260210_143022)
        self.run_id = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create output directory for this run
        # parents=True creates intermediate directories if needed
        # exist_ok=True prevents error if directory already exists
        self.output_dir = Path(f"outputs/{self.run_id}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Build the multi-agent team with termination conditions
        # This is done in __init__ so the team is ready when run_session is called
        self.team = self._create_team()

    def _create_team(self):
        """
        Create and configure the RoundRobinGroupChat team.

        RoundRobinGroupChat makes agents take turns in a fixed order:
        1. RE agent (asks question)
        2. User agent (responds)
        3. Back to RE agent
        ... repeat until termination condition is met

        Returns:
            RoundRobinGroupChat: Configured team ready to execute dialogue

        Termination Conditions:
            Two conditions combined with OR operator (|):
            1. MaxMessageTermination: Stops after max_turns exchanges
            2. TextMentionTermination: Stops when "REQUIREMENTS_COMPLETE" appears

            Whichever condition triggers first will end the conversation.

        Design Decision:
            RE agent is placed first in participants list to ensure it starts
            the conversation with the initial question.
        """
        # Combine RE agent and all user agents into participants list
        # Order matters: RE first ensures it gets the initial turn
        participants = [self.re_agent] + self.user_agents

        # Extract maximum conversation turns from scenario constraints
        # This prevents infinite loops and ensures sessions complete in reasonable time
        max_turns = self.scenario['constraints']['max_turns']

        # Conversation stops when EITHER condition is met
        termination = (
            MaxMessageTermination(max_messages=max_turns) |
            TextMentionTermination("REQUIREMENTS_COMPLETE")
        )

        # Create RoundRobinGroupChat team with configuration
        return RoundRobinGroupChat(
            participants=participants,
            termination_condition=termination
        )

    async def run_session(self, initial_message: str = None):
        """
        Execute a complete requirements elicitation session.

        This async method orchestrates the entire dialogue:
        1. Displays session header with run metadata
        2. Creates initial task message to start the conversation
        3. Streams messages as they are generated (for real-time monitoring)
        4. Detects conversation completion via TaskResult
        5. Saves dialogue transcript to JSON

        Args:
            initial_message (str, optional): Custom opening message to seed the session.
                If None, generates a default greeting based on the scenario name.

        Returns:
            TaskResult: AutoGen result object containing:
                - messages: List of all conversation messages
                - stop_reason: Why the conversation ended
            Returns None if the stream ends without a TaskResult.

        Streaming Pattern:
            Uses async for to process messages as they arrive, rather than
            waiting for the entire conversation to complete. This allows
            real-time console output and progress monitoring during development.

        Pattern Reference:
            1.-Understanding-MultiAgent-Systems.ipynb - Shows run_stream usage
            AgentChat_First_Agent.ipynb - Message handling patterns
        """
        # Generate default initial message if not provided.
        # This is a neutral system instruction that triggers the RE agent to open
        # the interview naturally in character, rather than a canned greeting.
        # The seed message itself is filtered out of the saved transcript in
        # _save_dialogue so the JSON starts with the RE agent's actual opening.
        if not initial_message:
            initial_message = "Begin the requirements elicitation interview."

        task = TextMessage(content=initial_message, source="user")

        print("\n--- DIALOGUE START ---\n")

        # Stream messages as they are generated by the team
        # run_stream returns an AsyncIterator that yields messages one by one
        async for message in self.team.run_stream(task=task):

            # Check if message is TaskResult — this signals the conversation has ended
            if isinstance(message, TaskResult):

                # Translate AutoGen stop reasons into readable messages
                if "REQUIREMENTS_COMPLETE" in (message.stop_reason or ""):
                    stop_display = "Session Terminated: Requirements Complete"
                else:
                    stop_display = "Session Terminated: Exceeded Maximum Turns Limit"

                print(f"\n--- DIALOGUE END ---")
                print(f"\n{stop_display}")
                print(f"Output: {self.output_dir}\n")

                # Save complete dialogue to JSON
                # message.messages contains all messages including the initial task
                self._save_dialogue(message.messages)

                return message
            else:
                # Regular message — print with double line break for readability
                # Skip the seed message (source="user") from console output
                if hasattr(message, 'source') and message.source != 'user':
                    print(f"{message.source}: {message.content}\n\n")

        # Return None if stream ends without TaskResult (should not occur in normal operation)
        return None

    def _save_dialogue(self, messages):
        """
        Persist the dialogue transcript to a JSON file for analysis.

        Saves the conversation in a structured format that can be read by
        extraction.py for requirement extraction and by evaluation.py for
        metric computation.

        Args:
            messages: List of message objects from TaskResult.messages.
                Each message may have 'source' and 'content' attributes.

        Side Effects:
            Creates/overwrites dialogue.json in the run's output directory.

        File Structure:
            {
                "run_id": "run_20260210_143022",
                "scenario": "Barbershop Booking System",
                "timestamp": "2026-02-10T14:30:45.123456",
                "total_messages": 12,
                "messages": [
                    {"source": "user", "content": "Hello..."},
                    {"source": "RE_experienced_methodical", "content": "What features..."},
                    {"source": "User_cooperative_expert", "content": "We need..."}
                ]
            }

        Design Decision:
            Uses hasattr checks for safe attribute access because some AutoGen message
            types may not have 'source' or 'content' attributes. Falls back to
            'unknown' and str(msg) respectively to ensure all messages are captured.
        """
        dialogue_data = {
            "run_id": self.run_id,
            "scenario": self.scenario['name'],
            "model": self.model_name or "unknown",
            "timestamp": datetime.now().isoformat(),
            "total_messages": len(messages),
            "messages": [
                {
                    "source": msg.source if hasattr(msg, 'source') else 'unknown',
                    "content": msg.content if hasattr(msg, 'content') else str(msg)
                }
                for msg in messages
                if not (hasattr(msg, 'source') and msg.source == 'user')
            ]
        }

        output_file = self.output_dir / "dialogue.json"

        with open(output_file, 'w', encoding="utf-8") as f:
            json.dump(dialogue_data, f, indent=2, ensure_ascii=False)

"""
Healthcare Multi-Agent Consultation System
Uses AutoGen's ConversableAgent and GroupChat for a real multi-agent pipeline.
"""

import logging
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("autogen.oai.client").setLevel(logging.ERROR)

from autogen import ConversableAgent, GroupChat, GroupChatManager


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LLM_CONFIG = {
    "config_list": [
        {
            "model": "gpt-4",
            "api_key": None,  # Set via OPENAI_API_KEY environment variable
        }
    ]
}

GROUP_CHAT_MAX_ROUNDS = 5


# ---------------------------------------------------------------------------
# Agent Definitions
# ---------------------------------------------------------------------------

def create_agents() -> dict[str, ConversableAgent]:
    """Create and return all agents used in the consultation pipeline."""

    patient_agent = ConversableAgent(
        name="patient",
        system_message=(
            "You are a patient describing your symptoms clearly and asking for medical help. "
            "Be specific about what you are experiencing."
        ),
        llm_config=LLM_CONFIG,
    )

    diagnosis_agent = ConversableAgent(
        name="diagnosis",
        system_message=(
            "You are a medical diagnosis assistant. "
            "Analyze the patient's symptoms and provide a concise possible diagnosis. "
            "Summarize your findings in one clear response."
        ),
        llm_config=LLM_CONFIG,
    )

    pharmacy_agent = ConversableAgent(
        name="pharmacy",
        system_message=(
            "You are a pharmacy assistant. "
            "Based on the diagnosis provided, recommend appropriate over-the-counter medications or treatments. "
            "Respond only once with your recommendation."
        ),
        llm_config=LLM_CONFIG,
    )

    consultation_agent = ConversableAgent(
        name="consultation",
        system_message=(
            "You are a medical consultation assistant. "
            "Review the diagnosis and pharmacy recommendation, then determine if an in-person doctor visit is required. "
            "Provide a final summary with a clear next step for the patient. "
            "IMPORTANT: Always end your response with 'Consultation Complete' to signal the end of the session."
        ),
        llm_config=LLM_CONFIG,
    )

    return {
        "patient": patient_agent,
        "diagnosis": diagnosis_agent,
        "pharmacy": pharmacy_agent,
        "consultation": consultation_agent,
    }


# ---------------------------------------------------------------------------
# Group Chat Setup
# ---------------------------------------------------------------------------

def create_group_chat(agents: dict[str, ConversableAgent]) -> GroupChatManager:
    """Set up the group chat with diagnosis, pharmacy, and consultation agents."""

    group_chat = GroupChat(
        agents=[
            agents["diagnosis"],
            agents["pharmacy"],
            agents["consultation"],
        ],
        messages=[],
        max_round=GROUP_CHAT_MAX_ROUNDS,
        speaker_selection_method="round_robin",
    )

    manager = GroupChatManager(
        name="manager",
        groupchat=group_chat,
    )

    return manager


# ---------------------------------------------------------------------------
# Consultation Runner
# ---------------------------------------------------------------------------

def run_consultation(symptoms: str) -> None:
    """Run the full multi-agent healthcare consultation for the given symptoms."""

    agents = create_agents()
    manager = create_group_chat(agents)

    opening_message = (
        f"I am feeling {symptoms}. "
        "Please help me with a diagnosis and recommendation for treatment."
    )

    agents["patient"].initiate_chat(
        manager,
        message=opening_message,
    )


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    print("\nWelcome to the AI Healthcare Consultation System!")
    print("This system uses multiple AI agents to assess your symptoms.\n")

    symptoms = input("Please describe your symptoms: ").strip()

    if not symptoms:
        print("No symptoms provided. Please try again.")
        return

    print("\nStarting consultation...\n")
    run_consultation(symptoms)


if __name__ == "__main__":
    main()
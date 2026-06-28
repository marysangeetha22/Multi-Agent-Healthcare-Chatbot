"""
Tests for healthcare_agents.py

Run with:
    pytest test_healthcare_agents.py -v

All AutoGen and LLM calls are mocked — no API key or network required.
"""

from unittest.mock import MagicMock, patch

import pytest

import healthcare_agents
from healthcare_agents import (
    GROUP_CHAT_MAX_ROUNDS,
    LLM_CONFIG,
    create_agents,
    create_group_chat,
    main,
    run_consultation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_conversable_agent():
    """Patch ConversableAgent so no LLM calls are made."""
    with patch("healthcare_agents.ConversableAgent") as MockAgent:
        MockAgent.side_effect = lambda name, **kwargs: MagicMock(name=name)
        yield MockAgent


@pytest.fixture
def mock_group_chat():
    """Patch GroupChat."""
    with patch("healthcare_agents.GroupChat") as MockGroupChat:
        yield MockGroupChat


@pytest.fixture
def mock_group_chat_manager():
    """Patch GroupChatManager."""
    with patch("healthcare_agents.GroupChatManager") as MockManager:
        yield MockManager


@pytest.fixture
def agents(mock_conversable_agent):
    """Return agents dict with mocked ConversableAgent instances."""
    return create_agents()


# ---------------------------------------------------------------------------
# Configuration Tests
# ---------------------------------------------------------------------------

class TestConfiguration:
    def test_llm_config_has_config_list(self):
        assert "config_list" in LLM_CONFIG

    def test_llm_config_model_is_gpt4(self):
        assert LLM_CONFIG["config_list"][0]["model"] == "gpt-4"

    def test_group_chat_max_rounds_is_positive(self):
        assert GROUP_CHAT_MAX_ROUNDS > 0

    def test_group_chat_max_rounds_value(self):
        assert GROUP_CHAT_MAX_ROUNDS == 5


# ---------------------------------------------------------------------------
# Agent Creation Tests
# ---------------------------------------------------------------------------

class TestCreateAgents:
    def test_returns_all_four_agents(self, agents):
        assert set(agents.keys()) == {"patient", "diagnosis", "pharmacy", "consultation"}

    def test_returns_dict(self, agents):
        assert isinstance(agents, dict)

    def test_all_values_are_mock_agents(self, agents):
        for agent in agents.values():
            assert agent is not None

    def test_agent_keys_are_strings(self, agents):
        for key in agents.keys():
            assert isinstance(key, str)

    def test_conversable_agent_called_four_times(self, mock_conversable_agent):
        create_agents()
        assert mock_conversable_agent.call_count == 4

    def test_patient_agent_created_with_correct_name(self, mock_conversable_agent):
        create_agents()
        names = [call.kwargs["name"] for call in mock_conversable_agent.call_args_list]
        assert "patient" in names

    def test_diagnosis_agent_created_with_correct_name(self, mock_conversable_agent):
        create_agents()
        names = [call.kwargs["name"] for call in mock_conversable_agent.call_args_list]
        assert "diagnosis" in names

    def test_pharmacy_agent_created_with_correct_name(self, mock_conversable_agent):
        create_agents()
        names = [call.kwargs["name"] for call in mock_conversable_agent.call_args_list]
        assert "pharmacy" in names

    def test_consultation_agent_created_with_correct_name(self, mock_conversable_agent):
        create_agents()
        names = [call.kwargs["name"] for call in mock_conversable_agent.call_args_list]
        assert "consultation" in names

    def test_all_agents_receive_llm_config(self, mock_conversable_agent):
        create_agents()
        for call in mock_conversable_agent.call_args_list:
            assert call.kwargs.get("llm_config") == LLM_CONFIG

    def test_consultation_system_message_contains_termination_signal(self, mock_conversable_agent):
        create_agents()
        consultation_call = next(
            c for c in mock_conversable_agent.call_args_list
            if c.kwargs.get("name") == "consultation"
        )
        assert "Consultation Complete" in consultation_call.kwargs["system_message"]


# ---------------------------------------------------------------------------
# Group Chat Tests
# ---------------------------------------------------------------------------

class TestCreateGroupChat:
    def test_returns_group_chat_manager(self, agents, mock_group_chat, mock_group_chat_manager):
        manager = create_group_chat(agents)
        assert manager is not None

    def test_group_chat_initialised_once(self, agents, mock_group_chat, mock_group_chat_manager):
        create_group_chat(agents)
        mock_group_chat.assert_called_once()

    def test_group_chat_manager_initialised_once(self, agents, mock_group_chat, mock_group_chat_manager):
        create_group_chat(agents)
        mock_group_chat_manager.assert_called_once()

    def test_group_chat_uses_correct_max_rounds(self, agents, mock_group_chat, mock_group_chat_manager):
        create_group_chat(agents)
        _, kwargs = mock_group_chat.call_args
        assert kwargs["max_round"] == GROUP_CHAT_MAX_ROUNDS

    def test_group_chat_uses_round_robin(self, agents, mock_group_chat, mock_group_chat_manager):
        create_group_chat(agents)
        _, kwargs = mock_group_chat.call_args
        assert kwargs["speaker_selection_method"] == "round_robin"

    def test_group_chat_excludes_patient_agent(self, agents, mock_group_chat, mock_group_chat_manager):
        create_group_chat(agents)
        _, kwargs = mock_group_chat.call_args
        agent_objects = kwargs["agents"]
        # Patient agent should not be in the group chat
        assert agents["patient"] not in agent_objects

    def test_group_chat_includes_three_agents(self, agents, mock_group_chat, mock_group_chat_manager):
        create_group_chat(agents)
        _, kwargs = mock_group_chat.call_args
        assert len(kwargs["agents"]) == 3

    def test_group_chat_starts_with_empty_messages(self, agents, mock_group_chat, mock_group_chat_manager):
        create_group_chat(agents)
        _, kwargs = mock_group_chat.call_args
        assert kwargs["messages"] == []

    def test_manager_named_correctly(self, agents, mock_group_chat, mock_group_chat_manager):
        create_group_chat(agents)
        _, kwargs = mock_group_chat_manager.call_args
        assert kwargs["name"] == "manager"


# ---------------------------------------------------------------------------
# run_consultation Tests
# ---------------------------------------------------------------------------

class TestRunConsultation:
    def test_initiates_chat(self, mock_conversable_agent, mock_group_chat, mock_group_chat_manager):
        run_consultation("headache")
        # patient agent's initiate_chat should be called
        patient_mock = mock_conversable_agent.return_value
        patient_mock.initiate_chat.assert_called_once()

    def test_message_contains_symptoms(self, mock_conversable_agent, mock_group_chat, mock_group_chat_manager):
        symptoms = "sore throat and fever"
        run_consultation(symptoms)
        patient_mock = mock_conversable_agent.return_value
        _, kwargs = patient_mock.initiate_chat.call_args
        assert symptoms in kwargs["message"]

    def test_message_is_a_string(self, mock_conversable_agent, mock_group_chat, mock_group_chat_manager):
        run_consultation("cough")
        patient_mock = mock_conversable_agent.return_value
        _, kwargs = patient_mock.initiate_chat.call_args
        assert isinstance(kwargs["message"], str)


# ---------------------------------------------------------------------------
# main() Tests
# ---------------------------------------------------------------------------

class TestMain:
    def test_main_calls_run_consultation_with_symptoms(self, mock_conversable_agent, mock_group_chat, mock_group_chat_manager):
        with patch("builtins.input", return_value="mild headache"):
            with patch("healthcare_agents.run_consultation") as mock_run:
                main()
                mock_run.assert_called_once_with("mild headache")

    def test_main_strips_whitespace_from_input(self, mock_conversable_agent, mock_group_chat, mock_group_chat_manager):
        with patch("builtins.input", return_value="  fever  "):
            with patch("healthcare_agents.run_consultation") as mock_run:
                main()
                mock_run.assert_called_once_with("fever")

    def test_main_does_not_call_run_consultation_on_empty_input(self, mock_conversable_agent, mock_group_chat, mock_group_chat_manager):
        with patch("builtins.input", return_value=""):
            with patch("healthcare_agents.run_consultation") as mock_run:
                main()
                mock_run.assert_not_called()

    def test_main_prints_welcome_message(self, mock_conversable_agent, mock_group_chat, mock_group_chat_manager, capsys):
        with patch("builtins.input", return_value=""):
            main()
        captured = capsys.readouterr()
        assert "Welcome" in captured.out

    def test_main_prints_error_on_empty_input(self, mock_conversable_agent, mock_group_chat, mock_group_chat_manager, capsys):
        with patch("builtins.input", return_value=""):
            main()
        captured = capsys.readouterr()
        assert "No symptoms provided" in captured.out
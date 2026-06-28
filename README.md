# AI Healthcare Consultation System

A multi-agent healthcare triage system built with [AutoGen](https://github.com/microsoft/autogen) and GPT-4. Four autonomous AI agents collaborate in a group chat to assess patient symptoms, suggest a diagnosis, recommend treatments, and determine whether an in-person doctor visit is needed.

---

## How It Works

The system runs a sequential multi-agent pipeline using AutoGen's `GroupChat` with round-robin speaker selection:

```
User Input (symptoms)
        │
        ▼
 Patient Agent          ← describes symptoms, initiates the chat
        │
        ▼
 Diagnosis Agent        ← analyses symptoms, provides a possible diagnosis
        │
        ▼
 Pharmacy Agent         ← recommends medications or treatments
        │
        ▼
 Consultation Agent     ← decides if a doctor visit is needed, gives final summary
```

Each agent has its own GPT-4 LLM instance and system prompt — this is real multi-agent reasoning, not hardcoded logic.

---

## Project Structure

```
├── healthcare_agents.py       # Main application
├── test_healthcare_agents.py  # Pytest test suite
└── README.md
```

---

## Requirements

- Python 3.10+
- An OpenAI API key with GPT-4 access

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/marysangeetha22/Multi-Agent-Healthcare-Chatbot.git
cd healthcare-agents
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Set your OpenAI API key**

```bash
export OPENAI_API_KEY=your-key-here        # macOS/Linux
set OPENAI_API_KEY=your-key-here           # Windows
```

---

## Usage

```bash
python healthcare_agents.py
```

You will be prompted to describe your symptoms:

```
Welcome to the AI Healthcare Consultation System!
This system uses multiple AI agents to assess your symptoms.

Please describe your symptoms: mild headache and fatigue

Starting consultation...
```

The agents then take turns in the group chat and produce a final consultation summary.

---

## Running Tests

No API key is needed to run the tests — all LLM and AutoGen calls are mocked.

```bash
pip install pytest
pytest test_healthcare_agents.py -v
```

Expected output:

```
test_healthcare_agents.py::TestConfiguration::test_llm_config_has_config_list          PASSED
test_healthcare_agents.py::TestConfiguration::test_llm_config_model_is_gpt4            PASSED
test_healthcare_agents.py::TestCreateAgents::test_returns_all_four_agents               PASSED
...
```

---

## Configuration

Both values are defined as constants at the top of `healthcare_agents.py` and can be changed without touching any other code:

| Constant | Default | Description |
|---|---|---|
| `LLM_CONFIG` | `gpt-4` | Model and API key configuration passed to all agents |
| `GROUP_CHAT_MAX_ROUNDS` | `5` | Maximum number of turns in the group chat |

---

## Disclaimer

This system is a demonstration project and is not a substitute for professional medical advice. Always consult a qualified healthcare provider for medical decisions.
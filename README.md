# AI Debate Platform

An intelligent debate platform where AI agents research topics and engage in structured debates with evidence-based arguments.

## Features

- **Research Phase**: Each debater conducts web research to gather evidence
- **Structured Debate**: Three perspectives (School, Student, Parent) debate topics
- **Evidence-Based Arguments**: All arguments must include citations and sources
- **Evaluation**: Neutral AI evaluator provides balanced recommendations

## How to Run

### Local Development
```bash
pip install -r requirements.txt
streamlit run streamlit_debate.py
```

### Command Line
```bash
python main.py --idea="Your debate topic" --n_round=6
```

## Deployment

This app is deployed on Streamlit Cloud. Visit the live version at: [Your App URL]

## Tech Stack

- **MetaGPT**: AI agent framework
- **Streamlit**: Web interface
- **Web Research**: Automated fact-gathering
- **Citation System**: Source verification
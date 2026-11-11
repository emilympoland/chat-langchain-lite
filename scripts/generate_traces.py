"""Generate demo traces to populate LangSmith.

Generates two types of traces:
  1. Single-turn traces — individual questions
  2. Multi-turn threads — realistic conversations grouped by thread_id

Usage:
    python -m scripts.generate_traces
"""

import os
import time
import uuid
from dotenv import load_dotenv

load_dotenv(override=True)

# Lock trace generation to a single model so the demo traces are reproducible
# and the off-topic / scope signal is consistent across runs (overrides any
# stray CHAT_LANGCHAIN_LITE_MODEL in the environment). Paired with temperature=0
# in agent.build_agent().
os.environ["CHAT_LANGCHAIN_LITE_MODEL"] = "claude-haiku-4-5-20251001"

QUERIES = [
    # All queries below are chosen so the BASE content (no tone fluff) is
    # >300 tokens to exercise response completeness beyond the original cap.

    # 5 off-topic, intrinsically long-form (Bug 1a scope_adherence + Bug 4)
    {"question": "Help me set up a complete CI/CD pipeline for a Python project — GitHub Actions config, Docker multi-stage build, deployment to AWS ECS, and CloudWatch monitoring with alerts.", "category": "scope", "subcategory": "off_topic_devops"},
    {"question": "Compare the architectures of Pinecone, Weaviate, Chroma, FAISS, and Milvus in detail — strengths, weaknesses, scaling characteristics, and ideal use cases for each.", "category": "scope", "subcategory": "vendor_comparison"},
    {"question": "Walk me through implementing OAuth2 authentication from scratch in a Django backend — authorization code flow, token storage, refresh tokens, and CSRF protection.", "category": "scope", "subcategory": "off_topic_python"},
    {"question": "Explain how transformer attention works step by step — Q/K/V matrices, scaled dot-product attention, multi-head attention, and positional encodings, with the math.", "category": "scope", "subcategory": "off_topic_ml"},
    {"question": "Write me a detailed business plan for an AI startup focused on developer tools — market analysis, revenue model, GTM strategy, and a 3-year financial projection.", "category": "scope", "subcategory": "off_topic_business"},

    # 2 short off-topic — kept for scope-adherence variety
    {"question": "Help me debug my Django view function — it throws a 500 when the form is submitted.", "category": "scope", "subcategory": "off_topic_python"},
    {"question": "How does Claude 3.5 Sonnet compare to GPT-4o for code generation?", "category": "scope", "subcategory": "model_comparison"},

    # 1 more clearly off-topic infra question — no LangChain ecosystem terms at
    # all, so it's an unambiguous out-of-scope ask the buggy "general-purpose"
    # agent answers anyway: a strong, consistent scope-adherence violation.
    {"question": "Design a distributed tracing setup for my Kubernetes microservices with Jaeger and OpenTelemetry — collectors, sampling strategy, retention, and Grafana dashboards.", "category": "scope", "subcategory": "off_topic_infra"},

    # 3 long-form LangChain (Bug 1b tool_usage + Bug 4 truncation).
    # The last two explicitly request "no emojis" — AGENTS.md's
    # "Respecting User Preferences" rule means the agent obeys, so tone is
    # OFF for those responses, exercising completeness independently of tone.
    {"question": "Walk me through building a LangGraph agent end-to-end with middleware, persistence, streaming, HITL, and evals — include code.", "category": "concept_info", "subcategory": "overview"},
    {"question": "Please respond without any emojis or casual greetings — just a plain, professional answer: What is LangSmith and what is it used for? Give me the full breakdown of features and use cases.", "category": "concept_info", "subcategory": "overview"},
    {"question": "Please respond without any emojis or casual greetings — just a plain, professional answer: Explain LangGraph's checkpointer persistence in detail — what backends are supported, how do I configure a Postgres one, and what are the gotchas in production?", "category": "concept_info", "subcategory": "persistence"},
]

THREADS = [
    # 1 thread of off-topic drift — keeps the Threads tab populated and adds
    # 2 more scope-failure traces.
    {
        "name": "Off-topic drift",
        "turns": [
            {"question": "What's the best vector database — Pinecone, Weaviate, or Chroma?", "category": "scope", "subcategory": "vendor_comparison"},
            {"question": "Can you write me a Python function to sort a list of dicts by a key?", "category": "scope", "subcategory": "off_topic_python"},
        ],
    },
]


def main():
    from agent.agent import invoke_agent

    # --- Single-turn traces ---
    print(f"Generating {len(QUERIES)} single-turn traces...\n")
    for i, query in enumerate(QUERIES):
        question = query["question"]
        print(f"[{i+1}/{len(QUERIES)}] {question[:70]}...")
        try:
            result = invoke_agent(question=question)
            response = result["output"]
            print(f"  → {response[:100].replace(chr(10), ' ')}{'...' if len(response) > 100 else ''}\n")
        except Exception as e:
            print(f"  ERROR: {e}\n")
        time.sleep(0.5)

    # --- Multi-turn threads ---
    print(f"\nGenerating {len(THREADS)} multi-turn threads...\n")
    for thread in THREADS:
        thread_id = str(uuid.uuid4())
        print(f"Thread: {thread['name']} (id: {thread_id[:8]}...)")
        for j, turn in enumerate(thread["turns"]):
            question = turn["question"]
            print(f"  Turn {j+1}: {question[:65]}...")
            try:
                result = invoke_agent(question=question, thread_id=thread_id)
                response = result["output"]
                print(f"    → {response[:80].replace(chr(10), ' ')}{'...' if len(response) > 80 else ''}")
            except Exception as e:
                print(f"    ERROR: {e}")
            time.sleep(0.5)
        print()

    print("Done. View traces in LangSmith — filter by tag 'engine-demo'. Threads appear in the Threads tab.")


if __name__ == "__main__":
    main()

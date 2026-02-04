# free_agentic_chatbot.py

import os
import operator
from typing import TypedDict, Annotated, Sequence, Literal
import json

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver


# ============================================================================
# STEP 1: Define Tools (Agent's Capabilities)
# ============================================================================

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the knowledge base for information about products, services, or FAQs.
    Use this when the user asks questions about your business.

    Args:
        query: The search query to find relevant information
    """
    # Simulated knowledge base - replace with your database
    knowledge = {
        "pricing": "Our pricing: Basic $29/month, Pro $99/month, Enterprise $299/month.",
        "features": "Features include AI analytics, real-time dashboards, integrations, 24/7 support.",
        "support": "Support: email support@company.com, live chat, phone 1-800-HELP (Mon-Fri 9AM-6PM EST).",
        "refund": "30-day money-back guarantee on all plans. Contact support for refunds.",
        "demo": "Schedule a free demo at any time. We offer personalized walkthroughs.",
        "trial": "Free 14-day trial available for all plans. No credit card required.",
        "integrations": "We integrate with Slack, Teams, Salesforce, HubSpot, and 100+ other tools."
    }

    # Simple keyword matching
    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return value

    return "I don't have specific information about that. Please contact our support team."


@tool
def calculate_estimate(users: int, plan: str = "pro") -> str:
    """
    Calculate a cost estimate based on number of users and plan type.

    Args:
        users: Number of users/seats needed
        plan: Plan type - basic, pro, or enterprise (default: pro)
    """
    pricing = {
        "basic": 29,
        "pro": 99,
        "enterprise": 299
    }

    base_price = pricing.get(plan.lower(), 99)

    # Volume discounts
    if users > 100:
        discount = 0.20
        total = base_price * users * (1 - discount)
        return f"For {users} users on {plan} plan: ${total:.2f}/month (20% volume discount applied)"
    elif users > 50:
        discount = 0.10
        total = base_price * users * (1 - discount)
        return f"For {users} users on {plan} plan: ${total:.2f}/month (10% volume discount applied)"
    else:
        total = base_price * users
        return f"For {users} users on {plan} plan: ${total:.2f}/month"


@tool
def web_search(query: str) -> str:
    """
    Search the web for current information.
    Use this when you need up-to-date information not in your knowledge base.

    Args:
        query: Search query
    """
    # Simulated web search - replace with real API (DuckDuckGo, SerpAPI, etc.)
    results = {
        "ai trends": "Current AI trends: Agentic AI, multimodal models, open-source LLMs gaining traction.",
        "python": "Python 3.12 is the latest stable version with improved performance and new features.",
        "market": "Tech market shows strong growth in AI/ML sectors with 45% YoY increase."
    }

    query_lower = query.lower()
    for key, value in results.items():
        if key in query_lower:
            return f"Search result: {value}"

    return f"Found information about: {query}. [Simulated search - integrate real API for production]"


@tool
def schedule_meeting(date: str, time: str, purpose: str) -> str:
    """
    Schedule a meeting or demo.

    Args:
        date: Preferred date (YYYY-MM-DD format)
        time: Preferred time (e.g., "2:00 PM")
        purpose: Meeting purpose
    """
    return f"✓ Meeting scheduled for {date} at {time}\nPurpose: {purpose}\nConfirmation email will be sent shortly."


@tool
def check_availability(service: str) -> str:
    """
    Check availability of services or features.

    Args:
        service: Service or feature name to check
    """
    services = {
        "api": "API access available on Pro and Enterprise plans",
        "sso": "Single Sign-On available on Enterprise plan only",
        "custom": "Custom integrations available on all plans with setup fee",
        "mobile": "Mobile apps available on iOS and Android for all plans"
    }

    service_lower = service.lower()
    for key, value in services.items():
        if key in service_lower:
            return value

    return f"Checking availability of {service}... Please contact sales for specific requirements."


# ============================================================================
# STEP 2: Define Agent State
# ============================================================================

class AgentState(TypedDict):
    """State that flows through the agent graph."""
    messages: Annotated[Sequence[BaseMessage], operator.add]


# ============================================================================
# STEP 3: Initialize FREE Local LLM (Ollama)
# ============================================================================

def create_llm(model_name: str = "gemma2:9b", temperature: float = 0.7):
    """
    Create a local LLM instance using Ollama.

    Recommended models:
    - llama3.1:8b (balanced, good for most tasks)
    - qwen3:8b (excellent for agentic tasks)
    - deepseek-r1:8b (best reasoning)
    - mistral:7b (fast)
    """
    print(f"🤖 Initializing {model_name} model...")

    llm = ChatOllama(
        model=model_name,
        temperature=temperature,
        base_url="http://localhost:11434",  # Default Ollama URL
    )

    return llm


# ============================================================================
# STEP 4: Tool Binding (Format for Ollama)
# ============================================================================

# Define all tools
tools = [
    search_knowledge_base,
    calculate_estimate,
    web_search,
    schedule_meeting,
    check_availability
]


def format_tools_for_prompt(tools_list):
    """Format tools as text for the system prompt."""
    tool_descriptions = []
    for tool in tools_list:
        tool_descriptions.append(
            f"- {tool.name}: {tool.description}"
        )
    return "\n".join(tool_descriptions)


# ============================================================================
# STEP 5: Agent Logic
# ============================================================================

SYSTEM_PROMPT = """You are a helpful AI customer service agent with access to tools.

Available Tools:
{tools}

Instructions:
1. Analyze the user's request carefully
2. Decide which tool(s) would be helpful
3. Call tools using this exact format:

TOOL_CALL: tool_name
ARGUMENTS: {{"arg1": "value1", "arg2": "value2"}}

4. You can call multiple tools if needed
5. After getting tool results, provide a helpful response to the user
6. Be conversational, friendly, and professional
7. If you don't need tools, just respond directly

Important: Always use the TOOL_CALL format exactly as shown above."""


def call_agent(state: AgentState, llm) -> AgentState:
    """Main agent reasoning node."""
    messages = state["messages"]

    # Prepare system prompt with tools
    system_msg = SYSTEM_PROMPT.format(tools=format_tools_for_prompt(tools))

    # Build conversation
    full_messages = [{"role": "system", "content": system_msg}]
    for msg in messages:
        if isinstance(msg, HumanMessage):
            full_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            full_messages.append({"role": "assistant", "content": msg.content})

    # Get LLM response
    response = llm.invoke(full_messages)

    return {"messages": [AIMessage(content=response.content)]}


def parse_tool_calls(text: str) -> list:
    """Parse tool calls from LLM response."""
    tool_calls = []

    if "TOOL_CALL:" not in text:
        return tool_calls

    lines = text.split("\n")
    current_tool = None
    current_args = None

    for line in lines:
        line = line.strip()

        if line.startswith("TOOL_CALL:"):
            current_tool = line.replace("TOOL_CALL:", "").strip()
        elif line.startswith("ARGUMENTS:"):
            args_str = line.replace("ARGUMENTS:", "").strip()
            try:
                current_args = json.loads(args_str)
            except:
                current_args = {}

            if current_tool:
                tool_calls.append({
                    "name": current_tool,
                    "arguments": current_args
                })
                current_tool = None
                current_args = None

    return tool_calls


def execute_tools(state: AgentState) -> AgentState:
    """Execute tools based on parsed tool calls."""
    messages = state["messages"]
    last_message = messages[-1]

    # Parse tool calls from last message
    tool_calls = parse_tool_calls(last_message.content)

    if not tool_calls:
        return {"messages": []}

    # Execute each tool
    tool_results = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["arguments"]

        # Find and execute the tool
        for tool in tools:
            if tool.name == tool_name:
                try:
                    result = tool.invoke(tool_args)
                    tool_results.append(f"[{tool_name} Result]: {result}")
                except Exception as e:
                    tool_results.append(f"[{tool_name} Error]: {str(e)}")
                break

    # Combine results
    combined_results = "\n\n".join(tool_results)

    return {"messages": [HumanMessage(content=f"Tool Results:\n{combined_results}")]}


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Decide whether to call tools or end."""
    messages = state["messages"]
    last_message = messages[-1]

    # Check if last message has tool calls
    if isinstance(last_message, AIMessage) and "TOOL_CALL:" in last_message.content:
        return "tools"

    return "end"


# ============================================================================
# STEP 6: Build Agent Graph
# ============================================================================

def create_agent_graph(model_name: str = "gemma2:9b"):
    """Create and compile the agent graph."""

    # Initialize LLM
    llm = create_llm(model_name)

    # Create graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", lambda state: call_agent(state, llm))
    workflow.add_node("tools", execute_tools)

    # Define flow
    workflow.add_edge(START, "agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )

    # After tools, go back to agent
    workflow.add_edge("tools", "agent")

    # Add memory
    memory = MemorySaver()

    # Compile
    agent_executor = workflow.compile(checkpointer=memory)

    return agent_executor


# ============================================================================
# STEP 7: Interactive Chat Interface
# ============================================================================

def run_chatbot(model_name: str = "gemma2:9b"):
    """Run the interactive chatbot."""

    print("=" * 70)
    print("🤖 FREE AGENTIC AI CHATBOT (Powered by Open-Source LLMs)")
    print("=" * 70)
    print(f"Using model: {model_name}")
    print("Type your message (or 'quit' to exit, 'clear' to reset conversation)")
    print("=" * 70)
    print()

    # Create agent
    agent = create_agent_graph(model_name)

    # Conversation config
    thread_id = "main-conversation"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        # Get user input
        user_input = input("You: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break

        if user_input.lower() == 'clear':
            thread_id = f"conversation-{os.urandom(4).hex()}"
            config = {"configurable": {"thread_id": thread_id}}
            print("✓ Conversation cleared!\n")
            continue

        if not user_input:
            continue

        # Process message
        try:
            messages = {"messages": [HumanMessage(content=user_input)]}

            print("\nAgent: ", end="", flush=True)

            response_text = ""
            for event in agent.stream(messages, config, stream_mode="values"):
                last_message = event["messages"][-1]

                if isinstance(last_message, AIMessage):
                    # Only print final response (not tool calls)
                    if "TOOL_CALL:" not in last_message.content:
                        response_text = last_message.content

            print(response_text)
            print()

        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Make sure Ollama is running: 'ollama serve'\n")


# ============================================================================
# STEP 8: Advanced RAG Agent (Optional - Using Free Embeddings)
# ============================================================================

def create_rag_agent():
    """
    Create an advanced RAG agent with free embeddings.
    Uses sentence-transformers (completely free, runs locally).
    """
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_core.documents import Document

    print("🔧 Setting up RAG with free embeddings...")

    # Create free local embeddings (no API key needed!)
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"  # Fast, free, runs locally
    )

    # Sample knowledge base documents
    documents = [
        Document(
            page_content="Our basic plan costs $29/month with 10 users, 50GB storage, email support.",
            metadata={"category": "pricing", "plan": "basic"}
        ),
        Document(
            page_content="Pro plan is $99/month: unlimited users, 500GB storage, priority support, API access.",
            metadata={"category": "pricing", "plan": "pro"}
        ),
        Document(
            page_content="Enterprise: $299/month with custom features, dedicated support, SLA guarantees, SSO.",
            metadata={"category": "pricing", "plan": "enterprise"}
        ),
        Document(
            page_content="AI analytics features: predictive modeling, anomaly detection, automated insights, custom dashboards.",
            metadata={"category": "features"}
        ),
        Document(
            page_content="Integrations available: Slack, Teams, Salesforce, HubSpot, Jira, GitHub, and 100+ tools.",
            metadata={"category": "integrations"}
        ),
        Document(
            page_content="Security: SOC 2 Type II certified, GDPR compliant, end-to-end encryption, regular audits.",
            metadata={"category": "security"}
        ),
    ]

    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    print("✓ RAG system ready with local embeddings!\n")

    # Create RAG tool
    @tool
    def semantic_search(query: str) -> str:
        """
        Search knowledge base using semantic similarity (RAG).
        Use for detailed questions about products, features, pricing, security.

        Args:
            query: Question or topic to search
        """
        docs = vectorstore.similarity_search(query, k=3)

        if not docs:
            return "No relevant information found."

        results = "\n\n".join([f"- {doc.page_content}" for doc in docs])
        return f"Relevant information:\n{results}"

    return semantic_search, embeddings


# ============================================================================
# STEP 9: Model Comparison Utility
# ============================================================================

def compare_models():
    """Compare different open-source models."""

    models = [
        ("llama3.1:8b", "Best overall, balanced performance"),
        ("qwen3:8b", "Excellent for agentic tasks"),
        ("deepseek-r1:8b", "Best reasoning capabilities"),
        ("mistral:7b", "Fastest, lightweight"),
        ("gemma2:9b", "Google's efficient model"),
    ]

    print("\n" + "=" * 70)
    print("📊 AVAILABLE FREE MODELS")
    print("=" * 70)

    for i, (model, description) in enumerate(models, 1):
        print(f"{i}. {model:<20} - {description}")

    print("\n💡 To switch models, restart with: python free_agentic_chatbot.py <model_name>")
    print("=" * 70 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import sys

    # Default model
    model = "gemma2:9b"

    # Allow model selection from command line
    if len(sys.argv) > 1:
        model = sys.argv[1]

    print("\n🚀 Starting Free Agentic AI Chatbot...")

    # Show available models
    # compare_models()

    # Optional: Setup RAG (uncomment to enable)
    # rag_tool, embeddings = create_rag_agent()
    # tools.append(rag_tool)

    # Run chatbot
    try:
        run_chatbot(model)
    except KeyboardInterrupt:
        print("\n\n👋 Chatbot stopped by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Ollama is running: 'ollama serve'")
        print(f"2. Make sure model is downloaded: 'ollama pull {model}'")
        print("3. Check if Ollama is at http://localhost:11434")

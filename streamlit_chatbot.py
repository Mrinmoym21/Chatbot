# streamlit_chatbot.py

import streamlit as st
from free_agentic_chatbot import create_agent_graph
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Free Agentic Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 Free Agentic AI Chatbot")
st.caption("Powered by Open-Source LLMs (Ollama + LangGraph)")

# Sidebar for model selection
model = st.sidebar.selectbox(
    "Select Model",
    ["gemma2:9b"]
)


# Initialize agent
@st.cache_resource
def get_agent(model_name):
    return create_agent_graph(model_name)


agent = get_agent(model)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit-conversation"

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        messages = {"messages": [HumanMessage(content=prompt)]}

        response_text = ""
        for event in agent.stream(messages, config, stream_mode="values"):
            last_message = event["messages"][-1]
            if isinstance(last_message, AIMessage) and "TOOL_CALL:" not in last_message.content:
                response_text = last_message.content
                message_placeholder.markdown(response_text)

        st.session_state.messages.append({"role": "assistant", "content": response_text})

# Clear button
if st.sidebar.button("Clear Conversation"):
    st.session_state.messages = []
    st.session_state.thread_id = f"conversation-{os.urandom(4).hex()}"
    st.rerun()

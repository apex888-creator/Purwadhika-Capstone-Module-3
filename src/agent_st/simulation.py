
from unittest import result

import streamlit as st
from langchain_core.messages import ToolMessage
from agent_st.agent import agent
from langfuse.langchain import CallbackHandler

langfuse_handler = CallbackHandler()


def send_chat(question: str, history: str) -> dict:
    messages_payload = history + [{"role": "user", "content": question}]
    result = agent.invoke(
        {"messages": messages_payload},
        config={"callbacks": [langfuse_handler]}
    )

    answer = result["messages"][-1].content

    total_input_tokens = 0
    total_output_tokens = 0

    for message in result["messages"]:
        if hasattr(message, "response_metadata") and message.response_metadata:
            if "usage_metadata" in message.response_metadata:
                total_input_tokens += message.response_metadata["usage_metadata"].get(
                    "input_tokens", 0)
                total_output_tokens += message.response_metadata["usage_metadata"].get(
                    "output_tokens", 0)
            elif "token_usage" in message.response_metadata:
                total_input_tokens += message.response_metadata["token_usage"].get(
                    "prompt_tokens", 0)
                total_output_tokens += message.response_metadata["token_usage"].get(
                    "completion_tokens", 0)

    price = 17_000 * (total_input_tokens * 0.10 +
                      total_output_tokens * 0.5) / 1_000_000

    tool_messages = []
    for message in result["messages"]:
        if isinstance(message, ToolMessage):
            tool_message_content = message.content
            tool_messages.append(tool_message_content)

    return {
        "answer": answer,
        "price": price,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "tool_messages": tool_messages
    }


st.title("🎬 Movies Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me anything about Movies!"):
    # Ambil 20 riwayat pesan terakhir untuk konteks
    messages_history = st.session_state.get("messages", [])[-20:]

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = send_chat(prompt, messages_history)
        answer = response["answer"]
        st.markdown(
            f"<div style='font-size: 14px; line-height: 1.5; white-space: pre-line;'>{answer}</div>", unsafe_allow_html=True)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer})

    with st.expander("**Tool Calls:**"):
        st.code(response["tool_messages"])

    with st.expander("**History Chat (JSON):**"):
        st.code(messages_history)

    with st.expander("**Usage Details:**"):
        st.code(
            f'Input Tokens: {response["total_input_tokens"]}\nOutput Tokens: {response["total_output_tokens"]}\nEstimated Price: Rp {response["price"]:.2f}')

from agent_st.agent import agent
import html
import json
import os
import sys
import re
import streamlit as st
from langchain_core.messages import ToolMessage
from langfuse.langchain import CallbackHandler

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))


langfuse_handler = CallbackHandler()

USD_TO_IDR = 17_000
INPUT_PRICE_PER_M = 0.10
OUTPUT_PRICE_PER_M = 0.5
MAX_FAILED_ATTEMPTS = 3
HISTORY_LIMIT = 20


def count_tokens(messages: list) -> tuple[int, int]:
    """Jumlahkan input/output token dari seluruh message hasil agent."""
    total_input, total_output = 0, 0

    for message in messages:
        metadata = getattr(message, "response_metadata", None)
        if not metadata:
            continue

        if "usage_metadata" in metadata:
            usage = metadata["usage_metadata"]
            total_input += usage.get("input_tokens", 0)
            total_output += usage.get("output_tokens", 0)
        elif "token_usage" in metadata:
            usage = metadata["token_usage"]
            total_input += usage.get("prompt_tokens", 0)
            total_output += usage.get("completion_tokens", 0)

    return total_input, total_output


def send_chat(question: str, history: list[dict]) -> dict:
    messages_payload = history + [{"role": "user", "content": question}]
    result = agent.invoke(
        {"messages": messages_payload},
        config={"callbacks": [langfuse_handler]},
    )

    answer = result["messages"][-1].content

    total_input_tokens, total_output_tokens = count_tokens(result["messages"])

    price = USD_TO_IDR * (
        total_input_tokens * INPUT_PRICE_PER_M
        + total_output_tokens * OUTPUT_PRICE_PER_M
    ) / 1_000_000

    tool_messages = [
        message.content
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    ]

    return {
        "answer": answer,
        "price": price,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "tool_messages": tool_messages,
    }


def render_message(role: str, content: str):
    """Render pesan dengan format yang sama untuk history dan jawaban baru."""
    if role == "assistant":
        cleaned = re.sub(r"\n\s*\n+", "\n", content.strip())
        safe_content = html.escape(cleaned)
        st.markdown(
            f"<div style='white-space: pre-line; line-height: 1.3;'>"
            f"{safe_content}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(content)


st.markdown(
    """
    <style>
    .stApp {
        background-color: #000000;
    }
    [data-testid="stBottom"], .stBottom {
        background-color: #000000 !important;
    }
    header, [data-testid="stHeader"] {
        background: transparent !important;
    }
 
    /* Menyamaratakan seluruh teks, paragraf, dan heading di dalam pesan bot menjadi ukuran normal (15px) */
    .stChatMessage div, .stChatMessage p, .stChatMessage h1, .stChatMessage h2, .stChatMessage h3, .stChatMessage span {
        font-size: 15px !important;
        font-weight: normal !important;
    }
 
    /* Merapatkan jarak antarbaris dan antar paragraf/list */
    .stChatMessage p, .stChatMessage li {
        margin-bottom: 0.2rem !important;
        line-height: 1.3 !important;
    }
    .stChatMessage ol, .stChatMessage ul {
        margin-bottom: 0.3rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True)

st.title("🎬 Movies Chatbot")

if "is_verified" not in st.session_state:
    st.session_state.is_verified = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "failed_attempts" not in st.session_state:
    st.session_state.failed_attempts = 0

if st.session_state.failed_attempts >= MAX_FAILED_ATTEMPTS:
    st.error("🚫 Access blocked! You have entered the wrong security code 3 times.")

elif not st.session_state.is_verified or not st.session_state.user_name:
    remaining_attempts = MAX_FAILED_ATTEMPTS - st.session_state.failed_attempts
    st.warning(
        f"⚠️ Please complete the verification and enter your name. "
        f"Remaining attempts: {remaining_attempts}"
    )

    with st.form("welcome_form"):
        st.write("**Security Verification:** What is the result of **8 + 5 - 6**?")
        user_answer = st.text_input("Math Answer:")

        st.write("**Profile:**")
        name_input = st.text_input("Enter your name:")

        submit_button = st.form_submit_button("Speak to Hercules")

        if submit_button:
            if user_answer.strip() != "7":
                st.session_state.failed_attempts += 1
                st.error(
                    f"Incorrect math answer! Failed attempts: "
                    f"{st.session_state.failed_attempts}/{MAX_FAILED_ATTEMPTS}"
                )
                st.rerun()
            elif not name_input.strip():
                st.error("Please enter your name.")
            else:
                st.session_state.is_verified = True
                st.session_state.user_name = name_input.strip()
                st.success("Welcome! Loading application...")
                st.rerun()

else:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": f"Hello {st.session_state.user_name}! I am Hercules, an AI assistant for the IMDb dataset. Please ask anything about movies, ratings, or comparisons!",
            }
        ]
    if st.session_state.messages[-1]["role"] == "user":
        st.session_state.messages.pop()
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            render_message(message["role"], message["content"])
    messages_history = st.session_state.get("messages", [])[-HISTORY_LIMIT:]

    if prompt := st.chat_input("Ask me anything about Movies!"):

        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Hercules still thinking..."):
                try:
                    response = send_chat(prompt, messages_history)
                except Exception as e:
                    response = {
                        "answer": f"Maaf, terjadi error: {e}",
                        "tool_messages": [], "total_input_tokens": 0,
                        "total_output_tokens": 0, "price": 0,
                    }

            answer = response["answer"]
            render_message("assistant", answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

            with st.expander("**Tool Calls:**"):
                st.code(
                    "\n\n".join(str(m) for m in response["tool_messages"])
                    or "(no tool calls)"
                )

            with st.expander("**History Chat (JSON):**"):
                st.code(
                    json.dumps(messages_history, indent=2, ensure_ascii=False),
                    language="json",
                )

            with st.expander("**Usage Details:**"):
                st.code(
                    f'Input Tokens: {response["total_input_tokens"]}\n'
                    f'Output Tokens: {response["total_output_tokens"]}\n'
                    f'Price: Rp {response["price"]:,.2f}'
                )

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
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎬 Movies Chatbot")

if "is_verified" not in st.session_state:
    st.session_state.is_verified = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "failed_attempts" not in st.session_state:
    st.session_state.failed_attempts = 0

# Check if the user is blocked due to 3 failed attempts
if st.session_state.failed_attempts >= 3:
    st.error("🚫 Access blocked! You have entered the wrong security code 3 times.")

elif not st.session_state.is_verified or not st.session_state.user_name:
    remaining_attempts = 3 - st.session_state.failed_attempts
    st.warning(
        f"⚠️ Please complete the verification and enter your name. Remaining attempts: {remaining_attempts}")

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
                    f"Incorrect math answer! Failed attempts: {st.session_state.failed_attempts}/3")
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
                "content": f"Hello {st.session_state.user_name}! I am Hercules, an AI assistant for the IMDb dataset. Please ask anything about movies, ratings, or comparisons!"
            }
        ]
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me anything about Movies!"):
        messages_history = st.session_state.get("messages", [])[-20:]

        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Hercules still thinking..."):
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

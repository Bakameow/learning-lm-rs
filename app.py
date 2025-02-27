import streamlit as st
import requests
import hashlib
from streamlit import success


def generate_hash(input_string: str, hash_algorithm: str = 'sha256') -> str:
    hash_func = hashlib.new(hash_algorithm)
    hash_func.update(input_string.encode('utf-8'))
    return hash_func.hexdigest()

def login_page():
    name = st.text_input("输入用户名")
    if name and st.button("确认"):
        st.session_state.logged_in = True
        if "user_id" not in st.session_state:
            st.session_state.user_id = generate_hash(name)

    if "messages" not in st.session_state:
        st.session_state.messages = []


def chat_page():
    # React to user input
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    global history
    if prompt := st.chat_input("向ChatBot发消息"):
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        data = {"session_id":st.session_state.user_id,
                "history":"",
                "system_message":"you are a helpful assistant.",
                "user_message":prompt}
        response = requests.post('http://localhost:8080/chat',json=data)
        with st.chat_message("assistant"):
            st.markdown(response.text)
        st.session_state.messages.append({"role": "assistant", "content": response.text})


def main():
    # 判断是否已登录
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if st.session_state.logged_in:
        # 如果已登录，显示聊天机器人界面
        chat_page()
    else:
        # 否则显示登录界面
        login_page()

if __name__ == "__main__":
    main()
import json
import os
import re

import gradio as gr
from http import HTTPStatus
import dashscope
from dashscope import Generation
from dashscope.api_entities.dashscope_response import Role
from typing import List, Optional, Tuple, Dict
from urllib.error import HTTPError
import time
from urllib.parse import quote
from RT_RAG import RT_RAG
from rag import RAG

default_system = 'You are a helpful assistant.'

# YOUR_API_TOKEN = os.getenv('YOUR_API_TOKEN')
# dashscope.api_key = YOUR_API_TOKEN

History = List[Tuple[str, str]]
Messages = List[Dict[str, str]]


def clear_session() -> History:
    rt_rag_system.clean_docs()
    return '', []


def modify_system_session(system: str) -> str:
    if system is None or len(system) == 0:
        system = default_system
    rt_rag_system.clean_docs()
    return system, system, []


def history_to_messages(history: History, system: str) -> Messages:
    messages = [{'role': Role.SYSTEM, 'content': system}]
    for h in history:
        messages.append({'role': Role.USER, 'content': h[0]})
        messages.append({'role': Role.ASSISTANT, 'content': h[1]})
    return messages


def messages_to_history(messages: Messages) -> Tuple[str, History]:
    assert messages[0]['role'] == Role.SYSTEM
    system = messages[0]['content']
    history = []
    for q, r in zip(messages[1::2], messages[2::2]):
        history.append([q['content'], r['content']])
    return system, history


def combine_history(history: History) -> str:
    combined_history = ""

    # 正则表达式模式，用于匹配 <details><summary> 及其后的所有内容
    details_pattern = re.compile(r'<details><summary>.*', re.DOTALL)

    for question, response in history:
        # 去除 <details><summary> 之后的内容
        cleaned_response = re.sub(details_pattern, '', response)
        # 拼接到 combined_responses 字符串中
        combined_history += question
        combined_history += cleaned_response
    return combined_history


def model_chat(query: Optional[str], history: Optional[History], system: str, search_urls=None
               ) -> Tuple[str, str, History]:
    if query is None:
        query = ''
    if history is None:
        history = []
    if search_urls is None:
        search_urls = []
    role = Role.SYSTEM
    response = ""
    messages = history_to_messages(history, system)
    messages.append({'role': Role.USER, 'content': query})
    if not history:
        combined_history = ""
        yield '', [[f'{query}', '查询重建...']], system
        new_query = rt_rag_system.query_reconstruction(query)
        yield '', [[f'{new_query}', '获取相关链接...']], system
        search_urls, titles = rt_rag_system.get_url_content(new_query)
        for (url, title) in zip(search_urls, titles):
            yield '', [[f'{new_query}', f"正在爬取: {title}-{url}"]], system
            time.sleep(0.1)
        yield '', [[f'{new_query}', '正在整理最相关的内容...']], system
        rt_rag_system.write(system)
    else:
        combined_history = combine_history(history[-3:])
    if not os.listdir(rt_rag_system.search_dir):
        response_words = rt_rag_system.direct_answer(query, combined_history)
    else:
        yield '', [[f'{query}', '稀疏检索...']], system
        filtered_documents_list = rt_rag_system.sparse_search(query)
        yield '', [[f'{query}', '文档编码（大约7秒）......']], system
        documents_vectors = rt_rag_system.embedding_transition(filtered_documents_list)
        yield '', [[f'{query}', '生成伪回答并编码（大约5秒）......']], system
        query_vectors = rt_rag_system.generate_Pseudo_answers(query, combined_history)
        yield '', [[f'{query}', '密集检索...']], system
        top_similar_docs = rt_rag_system.intensive_search(query_vectors, documents_vectors, filtered_documents_list)
        yield '', [[f'{query}', '生成最终答案...']], system
        response_words = rt_rag_system.get_answer(query, combined_history, top_similar_docs)
    for word in response_words:
        response += json.loads(word[6:])['data']['response'] + ""
        messages_to_yield = messages + [{'role': role, 'content': response.strip()}]
        system_state, history = messages_to_history(messages_to_yield)
        yield "", history, system_state
        time.sleep(0.02)
    if search_urls != []:
        formatted_links = [
            f"<a href='{url}'>{titles[i]}</a><br/>"
            for i, url in enumerate(search_urls)
        ]

        # 将格式化后的链接列表添加到response中
        response += "\n\n<details><summary>相关链接</summary>\n" + "\n".join(formatted_links) + "\n</details>"

    system, history = messages_to_history(messages + [{'role': role, 'content': response}])
    print('history', history)
    yield '', history, system


with gr.Blocks() as demo:
    with gr.Tab('新版'):
        gr.Markdown("""<center><font size=8>RT-RAG👾</center>""")
        rt_rag_system = RT_RAG()

        with gr.Row():
            with gr.Column(scale=3):
                system_input = gr.Textbox(value=default_system, lines=1, label='System')
            with gr.Column(scale=1):
                modify_system = gr.Button("🛠️ Set system prompt and clear history", scale=2)
            system_state = gr.Textbox(value=default_system, visible=False)
        chatbot = gr.Chatbot(label='RT-RAG')
        textbox = gr.Textbox(lines=1, label='Input')
        # 添加 HTML 组件显示加载动画
        # loading_spinner = gr.HTML('<div class="spinner"></div>', visible=False)

        with gr.Row():
            clear_history = gr.Button("🧹 Clear history")
            sumbit = gr.Button("🚀 Send")

        textbox.submit(model_chat,
                       inputs=[textbox, chatbot, system_state],
                       outputs=[textbox, chatbot, system_input])

        sumbit.click(model_chat,
                     inputs=[textbox, chatbot, system_state],
                     outputs=[textbox, chatbot, system_input],
                     concurrency_limit=5)
        clear_history.click(fn=clear_session,
                            inputs=[],
                            outputs=[textbox, chatbot])
        modify_system.click(fn=modify_system_session,
                            inputs=[system_input],
                            outputs=[system_state, system_input, chatbot])
    # with gr.Tab('旧版'):
    #     chatbot = gr.Chatbot()
    #     msg = gr.Textbox()
    #     clear = gr.Button("清除")
    #     file = gr.File(label='上传文本')
    #     rag_system = RAG()
    #
    #
    #     def rag_respond(message, chat_history):
    #         bot_message = rag_system.get_answer(message)
    #         chat_history.append((message, bot_message))
    #         time.sleep(1)
    #         # return "", chat_history
    #         return "", chat_history
    #
    #
    #     msg.submit(rag_respond, [msg, chatbot], [msg, chatbot])
    #     clear.click(lambda: None, None, chatbot, queue=False)
# demo.launch()
demo.launch(max_threads=5)
# demo.launch(max_threads=5, server_name="10.0.102.154")

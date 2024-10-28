from typing import Any

from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)

from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core import PromptTemplate
import requests
from llama_index.core import set_global_handler
set_global_handler('simple')


text_qa_template_str = (
    "Context information is"
    " below.\n---------------------\n{context_str}\n---------------------\nUsing"
    " both the context information and also using your own knowledge, answer"
    " the question: {query_str}\nIf the context isn't helpful, you can also"
    " answer the question on your own.\n"
)
text_qa_template = PromptTemplate(text_qa_template_str)

refine_template_str = (
    "The original question is as follows: {query_str}\nWe have provided an"
    " existing answer: {existing_answer}\nWe have the opportunity to refine"
    " the existing answer (only if needed) with some more context"
    " below.\n------------\n{context_msg}\n------------\nUsing both the new"
    " context and your own knowledge, update or repeat the existing answer.\n"
)
refine_template = PromptTemplate(refine_template_str)
def instruct(prompt):
    url = 'http://115.236.62.114:10013/v1/system_message_setting'
    data = {
        "messages": [],
        "prompt": prompt,
        'model': 'qwen1_5-14b-chat',
    }
    output = requests.post(url, json=data).json()
    Response = eval(output)['data']['response']

    return Response

class QwenCustomLLM(CustomLLM):
    context_window: int = 4096  # 上下文窗口大小
    num_output: int = 2048  # 输出的token数量
    model_name: str = "Qwen-1_4B-Chat"  # 模型名称
    # tokenizer: object = None  # 分词器
    # model: object = None  # 模型

    def __init__(self):
        super().__init__()
        # GPU方式加载模型
        #self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, device_map="cuda", trust_remote_code=True)
        #self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, device_map="cuda", trust_remote_code=True).eval()

        # CPU方式加载模型
        # self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, device_map="cpu", trust_remote_code=True)
        # self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, device_map="cpu", trust_remote_code=True)
        #self.model = self.model.float()

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        # 得到LLM的元数据
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()  # 回调函数
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        # 完成函数
        #print("完成函数")

        #inputs = self.tokenizer.encode(prompt, return_tensors='pt').cuda()  # GPU方式
        # inputs = self.tokenizer.encode(prompt, return_tensors='pt')  # CPU方式
        #outputs = self.model.generate(inputs, max_length=self.num_output)
        response = instruct(prompt)
        #print(response)
        return CompletionResponse(text=response)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        # 流式完成函数
        print("流式完成函数")

        # inputs = self.tokenizer.encode(prompt, return_tensors='pt').cuda()  # GPU方式
        # # inputs = self.tokenizer.encode(prompt, return_tensors='pt')  # CPU方式
        # outputs = self.model.generate(inputs, max_length=self.num_output)
        # response = self.tokenizer.decode(outputs[0])
        response = instruct(prompt)
        for token in response:
            yield CompletionResponse(text=token, delta=token)



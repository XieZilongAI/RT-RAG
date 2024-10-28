import concurrent
import json
import os
from concurrent.futures import ThreadPoolExecutor
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import SentenceTransformer

import dashscope
import numpy as np
import requests
import re
from collections import defaultdict, Counter
import math

from bert4vec import Bert4Vec
from bs4 import BeautifulSoup
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core import SimpleDirectoryReader
from sklearn.metrics.pairwise import cosine_similarity
import jieba
from chTokenizer import split_by_re
from time import time
from tqdm import tqdm
from urllib.parse import urlparse, parse_qs


# 更换成自己浏览器上的Cookie
class RT_RAG():
    def __init__(self, ):
        self.RN = 20
        self.PAGE = 3
        self.google_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
            'Accept-Encoding': 'gzip, deflate, br, zstd',
            'Cookie': 'HSID=Azt9QJDCJgTGlukHR; SSID=AdISmHJ_ROvOzr5LR; APISID=Y9A6Cr7Jc9lBmNQZ/AVreYWPKv9WDkzsuH; SAPISID=kcqGCO6cTsJpfe2_/ACsEeV6..........',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7'
        }

        self.baidu_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
            'Accept-Encoding': 'gzip, deflate, br, zstd',
            'Cookie': 'BIDUPSID=EFE51D5C560F2D574ED347D1DE875575; PSTM=1695346953; BAIDUID=E5D5DD1ECD27D8A8FCE231DBCB271296:FG=1; BD_UPN=12314753; BDUSS=mQ1cVg1QWtyWEF.........',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7'
        }
        self.embedding_model1 = BGEM3FlagModel('BAAI/bge-m3',use_fp16=True)
        self.embedding_model2 = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.llm_model_name = 'starling-lm-7b-alpha'
        self.llm_model_name1 = 'qwen1_5-14b-chat'
        # self.simbert_model = Bert4Vec(mode='paraphrase-multilingual-minilm')
        self.query_wrapper_prompt = [
            {"role": "system",
             "content": "you are a helpful assistant"}]
        # 加载停用词表
        self.stopwords = self.load_stopwords('process_word/stopwords.txt')
        self.search_dir = 'temp_search_docs'
        self.search_documents = []

    def get_baidu_url(self, query):
        # 按页获取网站url
        urls = []
        for i in range(1, self.PAGE):
            url = f"http://www.baidu.com/s?wd={query}&rn={self.RN}&pn={10 * i}"
            urls.append(url)
        # print(f'获取{len(urls)}页搜索内容')
        return urls

    def get_googel_url(self, query):
        urls = [f"https://www.google.com/search?q={query}"]
        for i in range(1, self.PAGE):
            start = i * 10
            urls.append(f"https://www.google.com/search?q={query}&start={start}")
        return urls

    # Helper function to perform a search using Baidu Search API
    def search_baidu(self, query):
        Baseurls = self.get_baidu_url(query)
        # 需要跳过的包含视频的链接关键词
        # video_keywords = ['video', 'youtube', 'v.qq', 'bilibili', 'haokan', 'iqiyi', ]

        link = set()
        # headers = {"apikey": BAIDU_API_KEY}
        for url in Baseurls:
            response = requests.get(url, headers=self.baidu_headers, allow_redirects=False)
            response.encoding = 'utf-8'  # 手动指定字符编码
            # 使用BeautifulSoup来解析HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            # print(soup)
            # 假设搜索结果的链接位于某个特定的HTML元素中，这里需要根据实际情况调整选择器
            results = soup.select('div#wrapper>div#wrapper_wrapper>div#container>div#content_left')
            # results=results.find_all('a')
            for note_element in results:
                new_note_element = note_element.prettify()
                soup = BeautifulSoup(new_note_element, "html.parser")
                li_elements = soup.find_all('a')
                for li in li_elements:
                    # final_url = get_final_url(li['href'])
                    # # 检查链接是否包含任何视频关键词
                    # if final_url is None or any(video_keyword in final_url for video_keyword in video_keywords):
                    #     # print("video:", final_url)
                    #     continue  # 如果包含，则跳过当前链接
                    href = li.get('href')
                    if href:
                        link.add(href)
                    else:
                        pass
        # print(f'共获取{len(link)}个相关链接')
        return link

    def search_selenium(self, query):
        Baseurls = self.get_baidu_url(query)
        link = set()

        return link

    def search_google(self, query):
        # base_urls = get_google_url(query)
        links = set()
        params = {
            'q': query,
            'num': 20,  # 请求更多的结果
            'start': 0,  # 从第一个结果开始
            'hl': 'en',
            'gl': 'us',
            'safe': 'off'
        }
        response = requests.get("https://www.google.com/search", headers=self.google_headers, params=params,
                                allow_redirects=False)
        # response = requests.get(f"https://www.google.com/search?q={query}", headers=headers, allow_redirects=False)
        response.encoding = 'utf-8'  # 手动指定字符编码
        soup = BeautifulSoup(response.text, 'html.parser')
        # print(soup)
        # Google搜索结果链接在 'div#search' 下的 'a' 标签中
        results = soup.select('div#main>div#cnt>div#rcnt>div#center_col')
        # results = soup.find_all('div', class_=re.compile(r'^yuRUbf$'))
        # print(len(results),results)
        for note_element in results:
            # print(note_element)
            new_note_element = note_element.prettify()
            soup = BeautifulSoup(new_note_element, "html.parser")
            li_elements = soup.find_all('a')
            # print(li_elements)
            for li in li_elements:
                href = li.get('ping')
                if href and href.startswith('/url?'):
                    parsed_url = urlparse(href)
                    query_params = parse_qs(parsed_url.query)
                    actual_url = query_params.get('url')
                    if actual_url:
                        links.add(actual_url[0])
        # print('length:',len(links))
        return links

    def clean_text(self, text):
        # 读取需要删除的模式
        with open('process_word/masked_words.txt', 'r', encoding='utf-8') as f:
            patterns_to_remove = [line.strip() for line in f if line.strip()]
        combined_pattern = '|'.join(f'(?:{pattern})' for pattern in patterns_to_remove)

        def remove_if_short(match):
            matched_text = match.group()
            if len(matched_text) <= 400:
                return ''
            else:
                return matched_text

        text = re.sub(combined_pattern, remove_if_short, text, flags=re.DOTALL)

        # 去除多余的空行和空白
        lines = text.splitlines()
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        return ''.join(cleaned_lines)

    def download(self, url):
        chars_to_skip = set("¢�®å")
        try:
            with requests.Session() as session:
                # now = time()
                response = session.get(url, headers=self.baidu_headers, allow_redirects=True, timeout=2)
                # print(f"下载耗时: {round(time() - now, 2)}s")

                # response = session.get(url, headers=self.google_headers, allow_redirects=False)
                final_url = response.url
                response.encoding = 'utf-8'
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    # print(soup)
                    text = self.clean_text(soup.get_text(separator=" ", strip=True))
                    if len(text) < 80 or any(char in chars_to_skip for char in text):
                        # print(f"Text from {url} is too short ({len(text)} characters), skipping.")
                        pass  # 跳过当前循环，不添加到documents列表
                    # 将清理后的文本添加到documents列表中
                    else:
                        title = soup.title.get_text()
                        # print("正常url：", url)
                        # print(text)
                        return {"text": text}, {"url": final_url}, {"title": title}
        except requests.exceptions.Timeout:
            # 请求超时直接返回空
            return None
        except Exception as e:
            return None

    # Helper function to download the content from the search results
    def download_content(self, urls):
        documents = []
        download_urls = []
        download_titles = []
        # print('downloading......')
        with ThreadPoolExecutor(max_workers=25) as executor:  # 可以调整max_workers以优化性能
            future_to_url = {executor.submit(self.download, url): url for url in urls}
            for future in tqdm(concurrent.futures.as_completed(future_to_url), total=len(urls)):
                result = future.result()
                if result is not None:  # 去重和检查返回结果
                    if result[2]["title"] not in download_titles:
                        download_urls.append(result[1]["url"])
                        download_titles.append(result[2]["title"])
                    if result[0]["text"] not in documents:
                        documents.append(result[0]["text"])
                        # Yield each result as it is processed
        return documents, download_urls, download_titles

    # 加载停用词表
    def load_stopwords(self, filepath):
        stopwords = set()
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                stopwords.add(line.strip())
        return stopwords

    # 对句子进行分词并去除停用词
    def preprocess_sentence(self, sentence):
        words = jieba.cut(sentence)
        filtered_words = [word for word in words if word not in self.stopwords and word.strip()]
        return filtered_words

    # 对文档列表进行预处理
    def preprocess_documents(self, docs):
        preprocessed_docs = []
        for doc in docs:
            preprocessed_doc = self.preprocess_sentence(doc)
            preprocessed_docs.append(preprocessed_doc)
        return preprocessed_docs

    # 构建倒排索引
    def build_inverted_index(self, docs):
        inverted_index = defaultdict(list)
        for doc_id, doc in enumerate(docs):
            for term in doc:
                inverted_index[term].append(doc_id)
        return inverted_index

    # 计算 TF-IDF 得分
    def compute_tfidf(self, docs, inverted_index):
        tfidf_scores = defaultdict(lambda: defaultdict(float))
        N = len(docs)

        # Compute document frequencies
        df = defaultdict(int)
        for term in inverted_index:
            df[term] = len(inverted_index[term])

        # Compute TF-IDF scores
        for doc_id, doc in enumerate(docs):
            # 计算词频
            tf = Counter(doc)
            max_tf = max(tf.values()) if tf else 0  # 防止除以零
            # 计算TF-IDF分数
            for term, freq in tf.items():
                df = len(inverted_index[term])  # 该词出现在多少个文档中
                if df > 0:
                    tfidf_scores[doc_id][term] = (freq / max_tf) * math.log(N / (df))
                else:
                    tfidf_scores[doc_id][term] = 0
        return tfidf_scores

    # 文档评分与排序
    def rank_documents(self, query, tfidf_scores, threshold=0.6):
        query_terms = list(jieba.cut(query))
        doc_scores = defaultdict(float)

        for doc_id, term_scores in tfidf_scores.items():
            for term in query_terms:
                if term in term_scores:
                    doc_scores[doc_id] += term_scores[term]

        # Filter out documents with scores below the threshold
        # filtered_doc_scores = {doc_id: score for doc_id, score in doc_scores.items() if score >= threshold}

        # Sort documents by score in descending order
        ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        # 然后获取前25%的结果
        num_top_results = min(int(len(ranked_docs)), 80)
        top_results = ranked_docs[:num_top_results]
        return top_results

    def jaccard_similarity(self, str1, str2):
        """计算两个字符串的Jaccard相似度"""
        a = set(str1.split())
        b = set(str2.split())
        return float(len(a.intersection(b))) / len(a.union(b))

    def get_context(self, query, documents_list, ranked_docs):
        context_length = 600  # 设定上下文长度

        # 用于存储带上下文的文档列表
        filtered_documents_list = []

        # 遍历每个排名的文档
        for doc_id, score in ranked_docs:
            current_text = documents_list[doc_id]  # 当前文档文本
            context = current_text  # 初始化上下文为当前文档文本
            i = 1  # 计数器用于从当前文档向前后扩展

            # 只要上下文长度小于context_length就继续添加
            while len(context) < context_length:
                # 尝试向后添加文本（优先向后添加两次）
                next_index = min(doc_id + i, len(documents_list) - 1)
                if next_index > doc_id:
                    next_text = documents_list[next_index]
                    # print('next_text1:', next_text)
                    # similarity = self.simbert_model.similarity(current_text, next_text, return_matrix=False)
                    # print('next_text1_sim:', similarity)
                    if next_text not in context:  # and similarity >= 0.3
                        temp_context = context + " " + next_text
                        if len(temp_context) > context_length:
                            break
                        context = temp_context
                        # 尝试再向后添加一次
                        next_index = min(doc_id + i + 1, len(documents_list) - 1)
                        if next_index > doc_id:
                            next_text = documents_list[next_index]
                            # print('next_text2:', next_text)
                            # similarity = self.simbert_model.similarity(current_text, next_text, return_matrix=False)
                            # print('next_text2_sim:', similarity)
                            if next_text not in context:
                                temp_context = context + " " + next_text
                                if len(temp_context) > context_length:
                                    break
                                context = temp_context

                # 尝试向前添加文本
                prev_index = max(doc_id - i, 0)
                if prev_index < doc_id:
                    prev_text = documents_list[prev_index]
                    # print('prev_text:', prev_text)
                    # similarity = self.simbert_model.similarity(current_text, prev_text, return_matrix=False)
                    # print('prev_text_sim:', similarity)
                    if prev_text not in context:
                        temp_context = prev_text + " " + context
                        if len(temp_context) > context_length:
                            break
                        context = temp_context

                i += 1  # 前后各扩展一次后计数器加1

            # 检查context是否已在列表中，或者与列表中的元素相似度过高
            is_similar = False
            if len(context) > context_length:
                context = context[:context_length]
            for j, existing_context in enumerate(filtered_documents_list):
                if context in existing_context:
                    is_similar = True
                    break  # 如果context已在列表中，跳过添加步骤
                if existing_context in context:
                    filtered_documents_list[j] = context  # 如果列表中的元素是context的子串，用context替换它
                    is_similar = True
                    break
                similarity = self.jaccard_similarity(context, existing_context)
                if similarity > 0.5:  # 设定相似度阈值
                    is_similar = True
                    break

            if not is_similar:  # 如果没有发现相似的context，则添加context到列表
                filtered_documents_list.append(context)

        return filtered_documents_list

    def check_format(self, response):
        # 将二进制数据转换成字符串
        data_str = response.content.decode('utf-8')
        # print(data_str)
        # 分割数据，以确保所有行都被正确处理
        lines = data_str.splitlines(keepends=True)

        # 初始化一个空列表来存储最终的段落
        paragraphs = []
        current_paragraph = ""

        # 正则表达式模式，匹配正确的行
        pattern = re.compile(r"^data: \{.*\}\r\n\r\n$", re.DOTALL)
        now = time()
        # 遍历所有的行
        for line in lines:
            current_paragraph += line
            # print(current_paragraph)
            # 检查当前段落是否符合正则表达式模式
            if pattern.match(current_paragraph):
                paragraphs.append(current_paragraph)
                current_paragraph = ""
            else:
                # 检查是否以 '\r\n\r\n' 结尾
                if current_paragraph.endswith('\r\n\r\n'):
                    paragraphs.append(current_paragraph)
                    current_paragraph = ""

        # 如果最后一个段落没有被添加，添加它
        if current_paragraph:
            paragraphs.append(current_paragraph)
        return paragraphs

    def query_reconstruction(self, query):
        message = [{'role': 'system',
                    "content": '''你是一个智能体，专门用于重建用户的查询以使其更加清晰和具体。请根据以下步骤重建查询：
                            1. 修正任何拼写错误。
                            2. 澄清任何模糊或不明确的表达。
                            3. 保持查询的原始意图和内容不变。
                            4. 不用回答查询本身的问题，只干以上三件事，也不用提醒用户该注意什么。
                            5. 当查询足够清晰和具体时，也没必要强加，返回原查询即可
                            以下是一些示例，展示了用户输入和重建后的查询：
                            示例1：
                            用户输入：大胜达近三年的营销情况
                            重建后的查询：大胜达纸包装公司2023，2022，2021年的营销情况。
                            示例2：
                            用户输入：苹果手机新款发布
                            重建后的查询：“Apple公司最新款iPhone的发布情况。”
                            示例3：
                            用户输入：北京天气
                            重建后的查询：中国北京今天的天气情况。
                            示例4：
                            用户输入：瓦楞纸箱的原材料有哪些？
                            重建后的查询：瓦楞纸箱的原材料有哪些？
                            要求：只生成一个最终重建结果，其他的内容一概不要输出，例如，”重建后的查询“，”请注意“，这些都不要。
                            注意：当前时间参考为2024年。
                            请根据上述原则，示例和要求，重建以下查询：'''}]

        pse_response = json.loads(requests.post('http://115.236.62.114:10013/v1/system_message_setting',
                                                json={'messages': message,
                                                      'model': self.llm_model_name,
                                                      'prompt': query,
                                                      'max_tokens': 64}).json())
        search_input = pse_response['data']['response']
        # pseudo_document = [query + '\n' + pse_response]
        print('新查询：', search_input)
        return search_input

    def get_url_content(self, query):
        # Perform web search
        search_urls = list(self.search_baidu(query))
        # print(search_urls)
        documents, final_urls, final_titles = self.download_content(search_urls)
        # print(documents)
        self.search_documents = documents
        return final_urls, final_titles

    def write(self, system_state):
        self.query_wrapper_prompt[0]['content'] = system_state
        # Load search documents into SimpleDirectoryReader format
        os.makedirs(self.search_dir, exist_ok=True)
        for i, doc in enumerate(self.search_documents):
            with open(os.path.join(self.search_dir, f'doc_{i}.txt'), 'w', encoding='utf-8') as f:
                f.write(doc)

    def direct_answer(self, query, history):
        prompt_query = "Based on the query given and historical information, write a high-quality answer for the query.\n query:" + query + "\nhistory:" + history

        try:
            response = requests.post('http://115.236.62.114:10013/v1/system_message_setting',
                                     json={'messages': self.query_wrapper_prompt,
                                           'model': self.llm_model_name,
                                           'prompt': prompt_query, 'stream': True},
                                     stream=True)

            # 打印响应
            # print("回答:", response['data']['response'])
            response = self.check_format(response)
            return response
        except requests.exceptions.RequestException as e:
            print("请求失败:", e)

    def sparse_search(self, query):
        # Read documents from the directory
        documents = SimpleDirectoryReader(self.search_dir).load_data(show_progress=True)
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
            sentence_splitter=split_by_re()
        )
        # 按照设置方法划分节点
        nodes = node_parser.get_nodes_from_documents(documents)
        documents_list = []
        for node in nodes:
            if len(node.text) < 10 or len(node.text) > 500:
                continue
            elif node.text not in documents_list:
                documents_list.append(node.text)

        print(len(documents_list), "个句子")
        for i, ju in enumerate(documents_list):
            print(i, ju)
        # 预处理文档
        preprocessed_docs = self.preprocess_documents(documents_list)
        # 构建倒排索引
        inverted_index = self.build_inverted_index(preprocessed_docs)

        # 计算TF-IDF分数
        tfidf_scores = self.compute_tfidf(preprocessed_docs, inverted_index)
        ranked_docs = self.rank_documents(query, tfidf_scores, threshold=0.6)
        filtered_documents_list = self.get_context(query, documents_list, ranked_docs)
        # yield '', [[f'{query}', '稀疏检索...']], system_state
        print("过滤后：", len(filtered_documents_list), "个句子")
        return filtered_documents_list

    def generate_Pseudo_answers(self, query, history):
        # 初始化一个空字符串来存储所有拼接后的回答
        now = time()
        message = [{'role': 'system',
                    "content": '''###初始化：
        你是一个很先进的智能体，专门用来根据给定的问题，写出与问题尽可能相关的高质量答案。如果你不能回答问题，请写一个简洁的、最有可能回答这个问题的答案模板。你的回答不得超过500字，所以不要包含任何与问题不相关的内容。注意：当前时间参考为2024年。
        ###系统限制：
        字数限制：回答不得超过500字。
        相关性：回答必须与用户的问题高度相关。
        简洁性：如果无法回答，提供一个简洁的答案模板。
        精确性：回答必须基于给定的问题，生成的所有内容都要与给定的问题有关。要避免引入说明式的信息，比如：“由于没有...，我将给出一个可能的回答框架：”。
        ###执行目标：
        1.生成高质量答案：在有把握回答时，基于问题，生成与问题高度相关的高质量答案。
        2.提供答案模板：在无法准确回答时，提供一个简洁的、最有可能的答案模板，其中你不知道的关键信息，用XXX表示。生成模板时不用另加说明，只生成答案模板。
        ###示例1：
        用户问题：“当前全球气候变化的主要原因是什么？”
        智能体回答：“2024年全球气候变化的主要原因是人类活动引起的碳排放、化石燃料使用和森林砍伐。”
        ###示例2：
        示例问题：“大胜达近三年营销情况”
        智能体回答：“大胜达纸包装公司近三年的销售情况如下：
        -2023年公司销售额为XXX亿元，与前一年相比变化了XX%。
        -2022年销售额达到XXX亿元，较上年增长XX%。
        -2021年，销售额为XXX亿元，同比增长XX%。”
        ###示例3：
        示例问题：“瓦楞纸箱的原材料”
        智能体回答：“瓦楞纸箱的原材料主要包括以下几种：
        面纸（表面层）：通常使用XXX材料，如XXX或XXX。
        芯纸（波纹芯层）：一般由XXX材料制成，如XXX或XXX。
        里纸（底层）：通常采用XXX材料，如XXX或XXX。
        粘合剂：用于粘合各层材料，通常使用XXX成分。“
        '''}]
        pse_response = json.loads(requests.post('http://115.236.62.114:10013/v1/system_message_setting',
                                                json={'messages': message,
                                                      'model': self.llm_model_name,
                                                      'prompt': query,
                                                      'max_tokens': 256}).json())
        pseudo_document = [query + '\n' + pse_response['data']['response']]
        # pseudo_document = [query + '\n' + pse_response]
        print(f"生成伪回答共耗时: {round(time() - now, 2)}s")  # 5.09s 9.14s 6.65s
        print('伪回答：', pseudo_document)
        # 把分好句的列表和问题进行embedding
        now = time()
        query_vectors = self.embedding_model2.encode(pseudo_document)
        # pse_embedding_response = requests.post('http://115.236.62.114:10013/v1/get_embedding',
        #                                        json={'prompt': pseudo_document}).json()
        # query_vectors = np.array(pse_embedding_response['data']['response']['data'][0]['embedding']).reshape(1, -1)
        print(f"伪回答和查询编码完毕共耗时: {round(time() - now, 2)}s")
        return query_vectors

    def embedding_transition(self, filtered_documents_list):
        if not filtered_documents_list:
            return None
        else:

            now = time()
            documents_vectors = self.embedding_model2.encode(filtered_documents_list)
            # doc_embedding_response = requests.post('http://115.236.62.114:10013/v1/get_embedding',
            #                                        json={'prompt': filtered_documents_list}).json()
            # new_embeddings = []
            # for i in range(len(doc_embedding_response['data']['response']['data'])):
            #     embedding = doc_embedding_response['data']['response']['data'][i]['embedding']
            #     if len(embedding) < 3584:
            #         embedding.extend([0] * (3584 - len(embedding)))
            #     new_embeddings.append(embedding)
            # # documents_vectors = np.array([documents_completion.data[i].embedding for i in range(len(documents_completion.data))])
            # documents_vectors = np.array(new_embeddings)
            print(f"documents_embedding完毕共耗时: {round(time() - now, 2)}s")
        return documents_vectors

    def intensive_search(self, query_vectors, documents_vectors, filtered_documents_list):
        if documents_vectors is None:
            return ""
        else:
            # 计算查询向量和文档库向量的点积相似度
            similarities = query_vectors @ documents_vectors.T

            # 找到相似度最高的10个索引
            top_indices = np.argsort(-similarities)[:, 0:10].reshape(-1)
            top_similar_docs = [filtered_documents_list[index] for index in top_indices]
        return top_similar_docs

    def get_answer(self, query, history, top_similar_docs):
        formatted_docs = "\n".join(f"Document [{i + 1}] {doc}" for i, doc in enumerate(top_similar_docs))
        prompt = "text documents:" + formatted_docs + "\nhistorical information:" + history + "\nUsing the above text documents and historical information (some of which might be irrelevant, and if none of these contexts are relevant to the query, answer in your own way.) write a brilliant answers around the given query. Do not disclose in your answers that you rely on these documents as if it were entirely your own knowledge.\n" + "query:" + query
        print(prompt)
        # Query index
        # print(self.query_wrapper_prompt)
        now = time()
        #
        response = requests.post('http://115.236.62.114:10013/v1/system_message_setting',
                                 json={'messages': self.query_wrapper_prompt,
                                       'model': self.llm_model_name,
                                       'prompt': prompt, 'stream': True}, stream=True)
        print(f"最终回答耗时: {round(time() - now, 2)}s")
        # now = time()
        response = self.check_format(response)
        # print(f"检查流式输出格式耗时: {round(time() - now, 2)}s")  # 5.82s 7.34s 9.02s
        return response

    def clean_docs(self):
        # Clean up temporary files
        for file_name in os.listdir(self.search_dir):
            file_path = os.path.join(self.search_dir, file_name)
            if os.path.isfile(file_path):
                os.unlink(file_path)

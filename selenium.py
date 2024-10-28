from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from time import time
import concurrent
import re
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

RN = 20
PAGE = 3

# 设置 Chrome 无头模式的选项
options = Options()
# options.add_argument('--headless')  # 无头模式
options.add_argument(
    'Mozilla/5.0 (iPhone; CPU iPhone OS 13_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/87.0.4280.77 Mobile/15E148 Safari/604.1')
options.add_argument('--disable-gpu')
options.add_argument('--disable-extensions')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--no-sandbox')
options.add_argument('--disable-images')
options.add_argument('--disable-javascript')
options.add_argument('--blink-settings=imagesEnabled=false')  # 禁用图片
options.add_argument('--disable-application-cache')  # 禁用缓存
options.add_argument('--disable-browser-side-navigation')  # 禁用侧边导航
options.add_argument('--disable-infobars')
options.add_argument('--disable-plugins-discovery')

# 使用 WebDriverManager 来自动管理 ChromeDriver
service = Service(ChromeDriverManager().install())

# 初始化 WebDriver 实例
driver = webdriver.Chrome(service=service, options=options)


def get_baidu_url(query):
    # 按页获取网站url
    urls = []
    for i in range(PAGE):
        url = f"http://www.baidu.com/s?wd={query}&rn={RN}&pn={10 * i}"
        urls.append(url)
    return urls


def search_baidu(query):
    Baseurls = get_baidu_url(query)
    link = set()
    for url in Baseurls:
        now = time()
        driver.get(url)
        print(f"解析页面耗时: {round(time() - now, 2)}s")
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        # print(soup)
        results = soup.select('div#wrapper>div#wrapper_wrapper>div#container>div#content_left')
        for note_element in results:
            new_note_element = note_element.prettify()
            soup = BeautifulSoup(new_note_element, "html.parser")
            li_elements = soup.find_all('a')
            for li in li_elements:
                href = li.get('href')
                if href:
                    link.add(href)
    print(f'共获取{len(link)}个相关链接')
    return link


def clean_text(text):
    # 通用的开头和结尾清理规则
    with open(' masked_words.txt', 'r', encoding='utf-8') as f:
        patterns_to_remove = [line.strip() for line in f if line.strip()]
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.DOTALL)
    # 去除多余的空行和空白
    lines = text.splitlines()
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    return ''.join(cleaned_lines)


def download(url, documents):
    try:
        # 设置页面加载的超时时间
        driver.set_page_load_timeout(2)  # 设置为2秒
        now = time()
        driver.get(url)
        print(f"下载一个url内容共耗时: {round(time() - now, 2)}s")
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        title = soup.title.get_text()
        # print(title)
        text = clean_text(soup.get_text(separator=" ", strip=True))
        if len(text) < 80 or text in documents:
            # print(f"Text from {url} is too short ({len(text)} characters), skipping.")
            pass  # 跳过当前循环，不添加到documents列表
        # 将清理后的文本添加到documents列表中
        else:
            # print("正常url：", url)
            return {"text": text}, {"url": url}, {"title": title}
    except Exception as e:
        print("页面加载超时，跳出循环")
        return None


def download_content(urls):
    documents = []
    download_urls = []
    download_titles = []
    with ThreadPoolExecutor(max_workers=1) as executor:  # 可以调整max_workers以优化性能
        future_to_url = {executor.submit(download, url, documents): url for url in urls}
        for future in tqdm(concurrent.futures.as_completed(future_to_url), total=len(urls)):
            result = future.result()
            if result is not None:  # 去重和检查返回结果
                if result[1]["url"] not in download_urls:
                    download_urls.append(result[1]["url"])
                    download_titles.append(result[2]["title"])
                if result[0]["text"] not in documents:
                    documents.append(result[0]["text"])
    return documents, download_urls, download_titles


query = "大胜达有限公司的营销情况。"
search_urls = list(search_baidu(query))
# print(search_urls)
now = time()
search_documents, final_urls, final_titles = download_content(search_urls)
print(f"下载内容共耗时: {round(time() - now, 2)}s")
driver.quit()

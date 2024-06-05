import schedule
import time
import os
import ast
import json
import pickle
import pandas as pd

# from datasets import load_dataset
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    BertTokenizer,
    BertModel,
    AdamW,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


class BERTEncoder(nn.Module):
    def __init__(self, bert_model_name):
        super(BERTEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.bert = BertModel.from_pretrained(bert_model_name)
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        # abstract_tokens = [tokenizer.decode(idx) for idx in abstract_tokens]
        # abstracts = " ".join(abstract_tokens)
        # inputs = self.tokenizer(abstracts, return_tensors="pt")
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # inputs = self.tokenizer("Hello world!", return_tensors="pt").to('cuda')
        # output_test = self.bert(**inputs)
        # print(output_test.last_hidden_state)
        return outputs.last_hidden_state[0]

    def one_embed(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.pooler_output

    def get_output_shape(self):
        return self.bert.config.hidden_size


class FFClassifier(nn.Module):
    # input_shape: bert output shape, num_classes = 3 (BIO)
    def __init__(self, input_shape, hidden_size, num_classes):
        super(FFClassifier, self).__init__()
        self.fc1 = nn.Linear(input_shape, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, token_embeds):
        # print(type(token_embeds))
        x = torch.relu(self.fc1(token_embeds))
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class Phraseformer(nn.Module):
    def __init__(
        self,
        bert_model_name,
        is_train_bert,
        is_graph_embedding,
        len_graph_embedding,
        path_graph_embedding,
    ):
        super(Phraseformer, self).__init__()
        # self.bert_model_name = bert_model_name
        self.bertEmbed = BERTEncoder(bert_model_name)
        self.len_graph_embedding = len_graph_embedding
        if is_train_bert:
            print("Có transfer learning bert")
        if is_graph_embedding:
            print("Có kết hợp graph embedding")
        self.ffclassifier = FFClassifier(
            self.bertEmbed.get_output_shape() + len_graph_embedding, 256, 3
        )

        # load graph_embedding
        with open(path_graph_embedding, "rb") as f:
            self.combined_vectors = pickle.load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, abstract_tokens, input_ids, attention_mask):
        bertEmbedding = self.bertEmbed(input_ids, attention_mask)
        # print("bertEmbedding.shape", bertEmbedding.shape)
        graphEmbedding = torch.tensor(
            [
                (
                    self.combined_vectors[abstract_token]
                    if abstract_token in self.combined_vectors
                    else [0] * self.len_graph_embedding
                )
                for abstract_token in abstract_tokens
            ]
        )
        add_padding = torch.tensor(
            [[-100] * self.len_graph_embedding] * (512 - graphEmbedding.shape[0])
        )
        # print("add_padding.shape", add_padding.shape)
        # print("graphEmbedding.shape", graphEmbedding.shape)
        graphEmbedding_pad = torch.cat((graphEmbedding, add_padding), dim=0).to(
            self.device
        )
        # print("graphEmbedding_pad.shape", graphEmbedding_pad.shape)
        full_embedding = torch.cat((bertEmbedding, graphEmbedding_pad), dim=1).to(
            torch.float32
        )
        # print(full_embedding.dtype)
        labels = self.ffclassifier(full_embedding)
        return labels

    def get_embedding(self, abstract_tokens, input_ids, attention_mask):
        bertEmbedding = self.bertEmbed.one_embed(input_ids, attention_mask)
        return bertEmbedding


# init hyperparameter of model
is_train_bert = True
is_graph_embedding = True
len_graph_embedding = 600
path_graph_embedding = "./combined_vectors_600_th3_1e5_full.pkl"
bert_model_name = "google-bert/bert-base-uncased"
max_length = 512
hidden_size = 256
num_classes = 3
batch_size = 1
num_epochs = 4
learning_rate = 2e-5

tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = Phraseformer(bert_model_name, is_train_bert, is_graph_embedding, len_graph_embedding, path_graph_embedding).to(device)

model_path = "./NO_BERT_WVFT_256_155_600_full.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))


# demo
max_length = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model_name = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

def tokenizerSplit(text):
  encoding = tokenizer(text, return_tensors='pt')
  decoding = [tokenizer.decode(idx) for idx in encoding['input_ids'][0]]
  return decoding[1:-1]

def extract_keywords_id(abstract_tokens, preds):
    keywords = []
    current_keyword = []
    for token, pred in zip(abstract_tokens, preds):
        if pred == 1:  # Nhãn thể hiện token bắt đầu một keyword
            if current_keyword != []:
                keywords.append(current_keyword)
            current_keyword = [token]  # Token đầu tiên của keyword
        elif pred == 2:  # Nhãn thể hiện token bên trong keyword
            current_keyword.append(token)  # Thêm token vào keyword
    if current_keyword != []:
        keywords.append(current_keyword)
    list_keyword = []
    for one_keyword in keywords:
        decoded_sequence = tokenizer.decode(one_keyword)
        list_keyword.append(decoded_sequence)
    return list_keyword

def find_keyword(abstract):
    # abstract = """A  bit  - serial  VLSI  neural  network  is  described  from  an  initial  architecture  for  a  synapse array through to silicon layout and board design.  The issues surrounding bit  - serial  computation,  and  analog/digital  arithmetic  are  discussed  and  the  parallel  development  of  a  hybrid  analog/digital  neural  network  is  outlined.  Learning  and  recall  capabilities  are  reported  for  the  bit  - serial  network  along  with  a  projected  specification  for  a  64  - neuron,  bit  - serial  board  operating  at 20 MHz.  This tech(cid:173) nique  is  extended  to  a  256  (2562  synapses)  network  with  an  update  time  of 3ms,  using  a  "paging"  technique  to  time  - multiplex  calculations  through  the  synapse  array."""
    abstract_tokens = tokenizerSplit(abstract)
    print(len(abstract_tokens))
    if (len(abstract_tokens)>max_length-2):
        return []
    encoding = tokenizer(abstract, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    # print(len(abstract_tokens))
    # print(input_ids.shape)
    # print(attention_mask.shape)
    outputs = model(abstract_tokens=abstract_tokens, input_ids=input_ids, attention_mask=attention_mask)
    outputs = outputs[1:len(abstract_tokens)+1]
    _, preds = torch.max(outputs, dim=1)
    preds_Id = extract_keywords_id(input_ids[0][1:len(abstract_tokens)+1], preds)
    unique_preds_Id = list(set(preds_Id))
    # print(unique_preds_Id)
    return unique_preds_Id

#############################################################################################
import csv
from selenium import webdriver
from bs4 import BeautifulSoup
import csv
import requests
import re
import os

driver = webdriver.Chrome()
base_url = 'https://proceedings.neurips.cc'

def extract_bibtex_content(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        bibtex_content = response.text
        return bibtex_content
    else:
        return None
        
def process_bibtex(bibtex_url, paper_info) :
    bibtex_url = base_url + bibtex_url
    bibtex_content = extract_bibtex_content(bibtex_url)

    pattern = r'\s*(.*?)\s*=\s*{(.*)}'
    matches = re.findall(pattern, bibtex_content)

    for key, value in matches:
        if key == 'author': paper_info['Authors'] = value.replace(',', '').replace(' and ', ', ')
        if key == 'booktitle' : paper_info['Book Title'] = value
        if key == 'editor': paper_info['Editors'] = value.replace(' and ', ', ')
        if key == 'pages' : paper_info['Pages'] = value
        if key == 'publisher' : paper_info['Publishers'] = value
        if key == 'title' : paper_info['Title'] = value
        if key == 'volume' : paper_info['Volume'] = value
        if key == 'year' : paper_info['Year'] = value
    return paper_info

def crawl_data(start = 1987, end = 2024):

    header = [
        'Year',
        'Volume',
        'Pages',
        'Status',
        'Book Title',
        'Title',
        'Authors',
        'Editors',
        'Publishers',
        'Main Url',
        'Metadata Url',
        'Paper Url',
        'Supplemental Url',
        'Review Url',
        'MetaReview Url',
        'AuthorFeedback Url',
        'Reviews And Public Comment',
        'Abstract'
    ]

    file_exists = os.path.isfile('papers_data.csv')
    if not file_exists:
        with open('papers_data.csv', mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
    
    with open('papers_data.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(header)

        ii = 0 
        for year in range(start, end):
            url = f"https://proceedings.neurips.cc/paper/{year}"
            driver.get(url)
            html_content = driver.page_source
            soup = BeautifulSoup(html_content, 'html.parser')

            paper_list_ul = soup.find('ul', class_='paper-list')
            conference_items = paper_list_ul.find_all('li')

            for item in conference_items:
                ii = ii + 1 

                paper_info = {key: None for key in header}

                if("conference" in item.get("class", [])): paper_info['Status'] = 'Main Conference Track'
                elif("datasets_and_benchmarks" in item.get("class", [])) : paper_info['Status'] = 'Datasets and Benchmarks Track'

                # main_url
                paper_info['Main Url'] = base_url + item.a['href'] 

                driver.get(paper_info['Main Url'])
                paper_html_content = driver.page_source
                paper_soup = BeautifulSoup(paper_html_content, 'html.parser')

                abstract_p_tags = paper_soup.select('div.container-fluid > div.col p')
                # Lấy nội dung của các thẻ <p> từ thứ 2 đến cuối cùng
                paper_info['Abstract'] = ' '.join([p_tag.text.strip() for p_tag in abstract_p_tags[2:]])

                div_bibtex_tags = paper_soup.select('div.container-fluid > div.col div')
                div_bibtex_first = div_bibtex_tags[0]
                a_tags = div_bibtex_first.find_all('a')
                for a_tag in a_tags:
                    href = a_tag.get('href')
                    text = a_tag.text
                    if text == 'Bibtex' : paper_info = process_bibtex(href, paper_info)
                    if text == 'Metadata' : paper_info['Metadata Url'] = base_url + href
                    if text == 'Paper' : paper_info['Paper Url'] = base_url + href
                    if text == 'Supplemental' : paper_info['Supplemental Url'] = base_url + href
                    if text == 'Review' : paper_info['Review Url'] = base_url + href # 2020 = Review . others = Reviews 
                    if text == 'Reviews' : paper_info['Review Url'] = base_url + href
                    if text == 'MetaReview' : paper_info['MetaReview Url'] = base_url + href
                    if text == 'AuthorFeedback' : paper_info['AuthorFeedback Url'] = base_url + href
                    if text == 'Reviews And Public Comment »' : paper_info['Reviews And Public Comment Url'] = base_url + href

                print(ii, ' - ', paper_info['Title'])
                writer.writerow([paper_info[key] for key in header])
        driver.quit()
    pass

def one_embed(abstract):
    # abstract = """A  bit  - serial  VLSI  neural  network  is  described  from  an  initial  architecture  for  a  synapse array through to silicon layout and board design.  The issues surrounding bit  - serial  computation,  and  analog/digital  arithmetic  are  discussed  and  the  parallel  development  of  a  hybrid  analog/digital  neural  network  is  outlined.  Learning  and  recall  capabilities  are  reported  for  the  bit  - serial  network  along  with  a  projected  specification  for  a  64  - neuron,  bit  - serial  board  operating  at 20 MHz.  This tech(cid:173) nique  is  extended  to  a  256  (2562  synapses)  network  with  an  update  time  of 3ms,  using  a  "paging"  technique  to  time  - multiplex  calculations  through  the  synapse  array."""
    abstract_tokens = tokenizerSplit(abstract)
    # print(len(abstract_tokens))
    if (len(abstract_tokens)>max_length-2):
        return []
    encoding = tokenizer(abstract, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    # print(len(abstract_tokens))
    # print(input_ids.shape)
    # print(attention_mask.shape)
    outputs = model.get_embedding(abstract_tokens=abstract_tokens, input_ids=input_ids, attention_mask=attention_mask)
    return outputs

import schedule
import time
from datetime import datetime
import ast
import csv
import torch


def daily_task():
    print("thuc hien daily_task")
    # Lấy năm hiện tại
    current_year = datetime.now().year
    # Lấy năm trước
    previous_year = current_year - 1
    # Kiểm tra nếu hôm nay là ngày đầu tiên của năm
    if datetime.now().strftime('%m-%d') == '01-01':
        print("Have crawl_data")
        crawl_data(previous_year, previous_year)
        
        # read file csv được crawl bởi mạnh
        df_test = pd.read_csv('./papers_data.csv')
        # Áp dụng hàm find_keyword cho mỗi hàng của DataFrame và tạo cột Keywords
        df_test['Keywords'] = df_test.apply(lambda x: find_keyword(x['Abstract']), axis=1)
        # Lưu DataFrame df_test thành một file CSV mới
        df_test.to_csv('./papers_data_done_new.csv', index=False)
        df_old = pd.read_csv('./papers_data_done.csv')
        result = pd.concat([df_old, df_test], axis=0)
        result.to_csv('./papers_data_done.csv', index=False)

        ####################################################################

        # Tạo một tập hợp để chứa tất cả các từ khóa
        all_keywords = set()

        # Lặp qua mỗi hàng dữ liệu trong cột 'Keywords'
        for keywords_list in df_test['Keywords']:
            print(keywords_list)
            # arr = ast.literal_eval(keywords_list)
            # Chuyển đổi mảng từ khóa thành tập hợp và thêm vào tập hợp chứa tất cả các từ khóa
            all_keywords.update(set(keywords_list))

        all_keywords_list = list(all_keywords)
        
        #
        one_emb_dict = {}
        for i in range(0, len(all_keywords_list), 1000):
            if (i+1000<len(all_keywords_list)):
                all_keywords_list_split = all_keywords_list[i:i+1000]
            else:
                all_keywords_list_split = all_keywords_list[i:]
            # can remove keyword here
            one_emb_dict_split = list(map(one_embed, all_keywords_list_split))
            for j in range(len(one_emb_dict_split)):
                one_emb_dict[all_keywords_list_split[j]] = one_emb_dict_split[j]
            print(i)

        print(len(one_emb_dict))

        # Giả sử từ điển có cấu trúc như {'key': tensor} và tensor có thể chuyển đổi thành list
        # Chuyển đổi dictionary thành danh sách để lưu vào file CSV

        csv_data = []

        for key, value in one_emb_dict.items():
            # Nếu giá trị là tensor, chuyển đổi nó thành list
            if isinstance(value, torch.Tensor):
                value = value.tolist()
            # Thêm key và value vào danh sách csv_data
            csv_data.append([key] + value)

        # Lưu vào file .csv
        with open("output_new.csv", "w", newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            # Ghi tiêu đề nếu cần (ví dụ: ['Key', 'Value1', 'Value2', ...])
            # csvwriter.writerow(['Key'] + ['Value' + str(i) for i in range(len(csv_data[0]) - 1)])
            # Ghi dữ liệu
            csvwriter.writerows(csv_data)

        # Hàm để đọc dữ liệu từ file CSV vào dictionary
        def read_csv_to_dict(file_path):
            data_dict = {}
            with open(file_path, mode='r') as csvfile:
                csvreader = csv.reader(csvfile)
                for row in csvreader:
                    key = row[0]
                    values = row[1:]
                    data_dict[key] = values
            return data_dict

        # Đọc dữ liệu từ hai file CSV
        new_data = read_csv_to_dict('output_new.csv')
        old_data = read_csv_to_dict('output.csv')

        # Hợp nhất dữ liệu, ưu tiên dữ liệu từ file mới
        merged_data = old_data.copy()
        merged_data.update(new_data)

        # Viết dữ liệu hợp nhất ra file CSV mới
        with open('output.csv', mode='w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            for key, values in merged_data.items():
                csvwriter.writerow([key] + values)

    else:
        print("Hôm nay không phải là ngày đầu năm. Tác vụ không được thực thi.")

# Đặt tác vụ để chạy mỗi ngày lúc 8 giờ sáng
schedule.every().day.at("08:00").do(daily_task)

# Vòng lặp để tiếp tục kiểm tra và chạy các tác vụ được lập lịch
while True:
    print("Auto Crawl Start 08:00 AM")
    schedule.run_pending()
    time.sleep(60)  # Ngủ trong 1 phút trước khi kiểm tra lại lịch


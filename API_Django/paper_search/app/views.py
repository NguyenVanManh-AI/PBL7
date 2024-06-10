from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.pagination import PageNumberPagination
import csv
import torch

PATH = './app/papers_data_done.csv'
PATH2 = './app/output.csv'
URL = './app/static/app/images/'

def search_by_keywords(keyword, csv_file_path):
    results = []
    with open(csv_file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Kiểm tra xem từ khóa có trong cột Keywords hay không
            if keyword.lower() in row['Keywords'].lower():
                # Nếu có, thêm hàng vào kết quả
                results.append(row)
    # Sắp xếp kết quả theo cột 'Year' giảm dần
    sorted_results = sorted(results, key=lambda x: int(x['Year']), reverse=True)
    return sorted_results

def search_by_all(keyword, csv_file_path):
    results = []
    with open(csv_file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Duyệt qua tất cả các cột trong hàng và kiểm tra từ khóa trong mỗi cột
            for column, value in row.items():
                if keyword.lower() in str(value).lower():
                    # Nếu có, thêm hàng vào kết quả và dừng vòng lặp cột
                    results.append(row)
                    break
    # Sắp xếp kết quả theo cột 'Year' giảm dần
    sorted_results = sorted(results, key=lambda x: int(x['Year']), reverse=True)
    return sorted_results

###################################################################################################
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
path_graph_embedding = "./app/combined_vectors_600_th3_1e5_full.pkl"
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

model_path = "./app/NO_BERT_WVFT_256_155_600_full.pth"
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



# abstract_demo = "this is my abstract. I want to extract keyword."
# print(find_keyword(abstract_demo))
# text = input('nhap to end:')

######################################################################################  


@api_view(['GET'])
def search(request):
    text = request.query_params.get('search', 'cnn')  # Lấy tham số search từ query params
    keywords = []
    search_by = "search_by_keywords"
    search_result = search_by_keywords(text, PATH)
    if search_result == []:
        search_by = "search_by_all"
        search_result = search_by_all(text, PATH)
    
    if search_result == []:
        search_by = "search_by_model"
        texts =find_keyword(text)
        keywords = texts
        print("trich xuat keyword:", texts)

        for text in texts:
            search_result.extend(search_by_keywords(text, PATH))

    if search_result == []:
        search_by = "case_not_result"
        search_result = search_by_keywords('cnn', PATH)
            
    # Tạo phân trang cho kết quả
    paginator = PageNumberPagination()
    paginated_search_result = paginator.paginate_queryset(search_result, request)

    # Tạo dictionary chứa thông tin phân trang và kết quả tìm kiếm
    response_data = {
        'search_by' : search_by,
        'keywords': keywords,
        'count': paginator.page.paginator.count,  # Tổng số mục
        'next': paginator.get_next_link(),        # Link tới trang kế tiếp (nếu có)
        'previous': paginator.get_previous_link(),# Link tới trang trước đó (nếu có)
        'results': paginated_search_result        # Kết quả tìm kiếm
    }

    # Trả về phản hồi RESTful API
    return Response(response_data, status=200)


class CustomPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100


import pandas as pd
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

@api_view(['GET'])
def trend_year(request):
    year = request.query_params.get('year', '') # Lấy tham số search từ query params
    if year == '':
        year = datetime.now().year - 1

    url, table = get_url_table(int(year))

    paginator = CustomPagination()
    result_page = paginator.paginate_queryset(table.to_dict('records'), request)
    response_data = {
        "url": "http://127.0.0.1:8000/static/app/images/wordcloud.png",
        "table": result_page,
    }    
    return paginator.get_paginated_response(response_data)

def get_url_table(year):
    df = pd.read_csv(PATH)  
    data_for_year = df[df['Year'] == year]
    dic = {}
    for data in data_for_year.Keywords:
        # Loại bỏ dấu ngoặc đơn và dấu phẩy ở đầu và cuối chuỗi
        string = data.strip("[]")

        # Tách chuỗi thành các từ dựa trên dấu phẩy và khoảng trắng
        words = string.split(", ")
        for word in words:
            if word[1:-1] not in dic:
                dic[word[1:-1]] = 1
            else:
                dic[word[1:-1]] += 1
    
    if "" in dic:
        dic[""] = 0

    if "re" in dic:
        dic["re"] = 0

    if "gith" in dic:
        dic["gith"] = 0
    
    if "ne" in dic:
        dic["ne"] = 0

    if "\\\\" in dic:
        dic["\\\\"] = 0

    if "ad" in dic:
        dic["ad"] = 0

    if "online" in dic:
        dic["online"] = 0

    # Chuyển dictionary thành DataFrame
    df_table = pd.DataFrame(dic.items(), columns=['Word', 'Frequency'])

    # Sắp xếp theo tần suất giảm dần
    df_table = df_table.sort_values(by='Frequency', ascending=False)

    # Lấy 20 từ xuất hiện nhiều nhất
    top_20_words = df_table

    # Tạo một dictionary từ DataFrame top_20_words
    word_freq_dict = dict(zip(top_20_words['Word'], top_20_words['Frequency']))
    
    # Tạo word cloud object
    wordcloud = WordCloud(width=800, height=400, background_color ='white').generate_from_frequencies(word_freq_dict)

    # Hiển thị word cloud
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    # Lưu sơ đồ wordcloud vào file ảnh
    image_path = URL + 'wordcloud.png'
    plt.savefig(image_path)
    plt.close()  # Đóng biểu đồ để giải phóng bộ nhớ
    return image_path, top_20_words
  
@api_view(['GET'])
def trend_10_year(request):
    year = request.query_params.get('year', '') # Lấy tham số search từ query params
    if year == '':
        year = datetime.now().year - 1
    else:
        year = int(year)
    
    # Đọc dữ liệu từ tệp CSV
    df = pd.read_csv(PATH)
    
    # Tạo một danh sách để lưu trữ dữ liệu từ khóa và tần suất
    keyword_data_list = []
    
    # Lặp qua từng năm từ 9 năm trước đến năm hiện tại
    for y in range(max(1987, year - 9), year + 1):
        # Lọc dữ liệu cho từng năm
        data_for_year = df[df['Year'] == y]
        
        # Tạo một từ điển để đếm tần suất của từng từ khóa cho năm hiện tại
        keyword_count = {}
        for keywords in data_for_year['Keywords']:
            # Xử lý chuỗi từ khóa và đếm tần suất xuất hiện của từng từ khóa
            words = keywords.strip('[]').split(', ')
            for word in words:
                word = word.strip("'")
                keyword_count[word] = keyword_count.get(word, 0) + 1

        if "" in keyword_count:
            keyword_count[""] = 0

        if "re" in keyword_count:
            keyword_count["re"] = 0

        if "gith" in keyword_count:
            keyword_count["gith"] = 0
        
        if "ne" in keyword_count:
            keyword_count["ne"] = 0

        if "\\\\" in keyword_count:
            keyword_count["\\\\"] = 0

        if "ad" in keyword_count:
            keyword_count["ad"] = 0

        if "online" in keyword_count:
            keyword_count["online"] = 0        
        # Thêm dữ liệu từ khóa và tần suất vào danh sách
        for keyword, freq in keyword_count.items():
            keyword_data_list.append({'Year': y, 'Keyword': keyword, 'Frequency': freq})
    
    # Tạo DataFrame từ danh sách dữ liệu từ khóa
    keyword_trends = pd.DataFrame(keyword_data_list)
    # Lấy top 10 từ khóa phổ biến nhất
    top_10_keywords = keyword_trends.groupby('Keyword').sum()['Frequency'].nlargest(10).index
    
    # Lọc dữ liệu cho top 10 từ khóa
    for keyword in top_10_keywords:
        keyword_data = keyword_trends[keyword_trends['Keyword'] == keyword]
        
        # Tạo biểu đồ đường cho từng từ khóa
        plt.plot(keyword_data['Year'], keyword_data['Frequency'], label=keyword)

    # Đặt tiêu đề và nhãn cho biểu đồ
    plt.title('Trend of Top 10 Keywords Over 10 Years')
    plt.xlabel('Year')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Lưu biểu đồ dưới dạng ảnh
    image_path = URL + 'line_graph.png'
    plt.savefig(image_path)
    plt.close()  # Đóng biểu đồ để giải phóng bộ nhớ
    
    response_data = {
        "url": "http://127.0.0.1:8000/static/app/images/line_graph.png",
    }    
    # Trả về phản hồi RESTful API
    return Response(response_data, status=200)



def plot_keyword_trends(keywords, csv_path, end_year):
    # Đọc dữ liệu từ tệp CSV
    df = pd.read_csv(csv_path)
    
    # Tạo một danh sách để lưu trữ dữ liệu từ khóa và tần suất
    keyword_data_list = []
    
    # Lặp qua từng năm từ 9 năm trước đến năm hiện tại
    for y in range(max(1987, end_year - 9), end_year + 1):
        # Lọc dữ liệu cho từng năm
        data_for_year = df[df['Year'] == y]
        
        # Tạo một từ điển để đếm tần suất của từng từ khóa cho năm hiện tại
        keyword_count = {keyword: 0 for keyword in keywords}
        for keywords_row in data_for_year['Keywords']:
            # Xử lý chuỗi từ khóa và đếm tần suất xuất hiện của từng từ khóa
            words = keywords_row.strip('[]').split(', ')
            for word in words:
                word = word.strip("'")
                if word in keyword_count:
                    keyword_count[word] += 1
        
        if "" in keyword_count:
            keyword_count[""] = 0

        if "re" in keyword_count:
            keyword_count["re"] = 0

        if "gith" in keyword_count:
            keyword_count["gith"] = 0
        
        if "ne" in keyword_count:
            keyword_count["ne"] = 0

        if "\\\\" in keyword_count:
            keyword_count["\\\\"] = 0

        if "ad" in keyword_count:
            keyword_count["ad"] = 0

        if "online" in keyword_count:
            keyword_count["online"] = 0   
        # Thêm dữ liệu từ khóa và tần suất vào danh sách
        for keyword, freq in keyword_count.items():
            keyword_data_list.append({'Year': y, 'Keyword': keyword, 'Frequency': freq})
    
    # Tạo DataFrame từ danh sách dữ liệu từ khóa
    keyword_trends = pd.DataFrame(keyword_data_list)
    
    # Vẽ biểu đồ đường cho các từ khóa
    plt.figure(figsize=(14, 8))
    for keyword in keywords:
        keyword_data = keyword_trends[keyword_trends['Keyword'] == keyword]
        plt.plot(keyword_data['Year'], keyword_data['Frequency'], marker='o', label=keyword)
    
    # Đặt tiêu đề và nhãn cho biểu đồ
    plt.title('Trend of Keywords Over 10 Years')
    plt.xlabel('Year')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)

    # Lưu biểu đồ dưới dạng ảnh
    image_path = URL + 'line_graph_2.png'
    plt.savefig(image_path)
    plt.close()  # Đóng biểu đồ để giải phóng bộ nhớ
    


##############################################################################
import torch
import csv
import ast
import numpy as np

reconstructed_dict = {}

# with open("./app/output.csv", "r", encoding='utf-8') as csvfile:
#     csvreader = csv.reader(csvfile)
#     for row in csvreader:
#         key = row[0]
#         # print(row[1:][0])
#         # Chuyển đổi giá trị từ danh sách sang tensor
#         row_data = ast.literal_eval(row[1:][0])
#         values = list(map(float, row_data))  # Assuming the values were floats
#         reconstructed_dict[key] = torch.tensor(values)

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def top_10(keyword, reconstructed_dict):

  embed_keyword = reconstructed_dict[keyword]
  distance_arrs = {}
  for key, value in reconstructed_dict.items():
    if key == "":
        continue

    if key == "re":
        continue

    if key == "gith":
        continue

    if key == "ne":
        continue

    if key == "\\\\":
        continue

    if key == "ad":
        continue

    if key == "online":
        continue
    distance_arrs[key] = euclidean_distance(embed_keyword, value)
    # print(distance_arrs[key])
  sorted_dict = dict(sorted(distance_arrs.items(), key=lambda item: item[1]))
  # Lấy top 10 phần tử đầu tiên
  top_10 = dict(list(sorted_dict.items())[:10])

  print(top_10)
  return list(top_10.keys()), list(top_10.values())

@api_view(['GET'])
def trend_10_keywords(request):
    year = request.query_params.get('year', '') # Lấy tham số search từ query params
    keyword = request.query_params.get('keyword', '')
    if year == '':
        year = datetime.now().year - 1
    else:
        year = int(year)

    keywords, euclid = top_10(keyword, reconstructed_dict)
    plot_keyword_trends(keywords, PATH, year)
    
    response_data = {
        "url": "http://127.0.0.1:8000/static/app/images/line_graph_2.png",
        "keywords": keywords,
        "euclid": euclid
    }    
    # Trả về phản hồi RESTful API
    return Response(response_data, status=200)  


####################################################################
#API tracking
PATH_TRACKING = './app/tracking.csv'
import csv

def add_to_tracking_csv(id_paper, keywords):
    # Đọc dữ liệu từ file tracking.csv
    rows = []
    try:
        with open(PATH_TRACKING, 'r') as file:
            reader = csv.reader(file)
            rows = list(reader)
    except FileNotFoundError:
        # Nếu file không tồn tại, tạo header cho file mới
        rows = [['id_paper', 'keywords']]

    # Kiểm tra xem id_paper đã tồn tại trong file hay chưa
    existing_row = None
    for row in rows[1:]:
        if row[0] == id_paper:
            existing_row = row
            break

    if existing_row:
        # Nếu id_paper đã tồn tại, thêm những keywords chưa tồn tại vào
        existing_keywords = existing_row[1].split(',')
        new_keywords = [keyword for keyword in keywords if keyword not in existing_keywords]
        if new_keywords:
            updated_keywords = existing_row[1] + ',' + ','.join(new_keywords)
            existing_row[1] = updated_keywords.strip(',')
    else:
        # Nếu id_paper chưa tồn tại, thêm một hàng mới vào file
        new_row = [id_paper, ','.join(keywords)]
        rows.append(new_row)

    # Ghi dữ liệu đã cập nhật vào file tracking.csv
    with open(PATH_TRACKING, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

@api_view(['POST'])
def tracking(request):
    try:
        # Parse the JSON payload from the request body
        data = request.data
        id_paper = data.get('id_paper', '')
        keywords = data.get('keywords', [])
        
        if not id_paper or not isinstance(keywords, list):
            return Response({"error": "Invalid input data"}, status=400)

        # Sử dụng hàm add_to_tracking_csv
        add_to_tracking_csv(id_paper, keywords)    

        response_data = {
            "id_paper": id_paper,
            "keywords": keywords,
            "status": "Đã thêm vào tracking.csv",
        }    
        return Response(response_data, status=200)
    
    except Exception as e:
        return Response({"error": str(e)}, status=500)
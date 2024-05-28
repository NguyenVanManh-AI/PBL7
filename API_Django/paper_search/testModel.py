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

abstract_demo = "this is my abstract. I want to extract keyword."
print(find_keyword(abstract_demo))
text = input('nhap to end:')



# CUDA Version: 12.2  
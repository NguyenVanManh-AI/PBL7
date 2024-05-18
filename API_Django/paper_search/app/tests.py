
import torch
import csv
import ast
import numpy as np

reconstructed_dict = {}


##############################################################################
# with open("./app/output.csv", "r", encoding='utf-8') as csvfile:
#     csvreader = csv.reader(csvfile)
#     for row in csvreader:
#         key = row[0]
#         # print(row[1:][0])
#         # Chuyển đổi giá trị từ danh sách sang tensor
#         row_data = ast.literal_eval(row[1:][0])
#         values = list(map(float, row_data))  # Assuming the values were floats
#         reconstructed_dict[key] = torch.tensor(values)
##########################################################################


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def top_10(keyword, reconstructed_dict):
  
  embed_keyword = reconstructed_dict[keyword]
  distance_arrs = {}
  for key, value in reconstructed_dict.items():
    distance_arrs[key] = euclidean_distance(embed_keyword, value)
    # print(distance_arrs[key])
  sorted_dict = dict(sorted(distance_arrs.items(), key=lambda item: item[1]))
  # Lấy top 10 phần tử đầu tiên
  top_10 = dict(list(sorted_dict.items())[:10])

  print(top_10)


# truyền vào một keyword muốn tìm 10 keyword liên quan nhất vd "cnn"
keyword = 'transformer'
import datetime as datetime
print("Time Begin: ", datetime.datetime.now())
top_10(keyword, reconstructed_dict)
print("Time End: ", datetime.datetime.now())

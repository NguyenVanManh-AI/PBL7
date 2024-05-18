import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Đường dẫn tệp CSV
PATH = './app/papers_data_done.csv'

# Dữ liệu các từ khóa và giá trị tương đồng
keywords = [
    'transformer', 'vitae transformer', 'adapter', '3dversarial generator', 
    'quadattack', 'retvecbedding model', 'beamrecuionly', 
    'efint algorithm', 'noisy power methodm', 'collaborative transformer'
]

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
    
    # Hiển thị biểu đồ
    plt.show()

# Gọi hàm để vẽ biểu đồ xu hướng cho các từ khóa từ 10 năm trước tới năm y
end_year = datetime.now().year - 1  # Ví dụ: năm hiện tại trừ 1
plot_keyword_trends(keywords, PATH, end_year)

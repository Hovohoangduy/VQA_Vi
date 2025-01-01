import json
import os
import pandas as pd

# Đọc tất cả các file JSON trong thư mục
input_dir = '/Users/duyhoang/Documents/Research/VQA/VQA_Vi/json'  # Thay thế bằng đường dẫn đến thư mục chứa các file JSON
output_dir = '/Users/duyhoang/Documents/Research/VQA/VQA_Vi/csv'  # Thay thế bằng đường dẫn đến thư mục lưu các file CSV

# Kiểm tra nếu thư mục output không tồn tại thì tạo nó
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Lặp qua tất cả các file trong thư mục
for file_name in os.listdir(input_dir):
    if file_name.endswith('.json'):  # Chỉ xử lý các file JSON
        json_file_path = os.path.join(input_dir, file_name)
        
        # Đọc dữ liệu JSON từ file
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Danh sách lưu các dòng dữ liệu
        rows = []

        # Lặp qua từng annotation để lấy dữ liệu
        for anno in data["annotations"]:
            anno_id = anno["id"]
            image = f"{anno['image_id']}.jpg"  # Đổi image_id thành image.jpg
            question = anno["question"]
            answer = ", ".join(anno["answers"])  # Kết nối các câu trả lời nếu có nhiều hơn một
            
            rows.append([anno_id, image, question, answer])

        # Tạo DataFrame từ dữ liệu
        df = pd.DataFrame(rows, columns=["anno_id", "image", "question", "answer"])

        # Lưu DataFrame vào file CSV, lấy tên file JSON và đổi đuôi thành .csv
        csv_file_name = os.path.splitext(file_name)[0] + '.csv'
        csv_file_path = os.path.join(output_dir, csv_file_name)
        
        df.to_csv(csv_file_path, index=False)
        
        print(f"CSV file '{csv_file_name}' has been created successfully.")
import csv
import os

def delete_first_two_rows(csv_file):
    temp_file = csv_file + ".tmp"  # 임시 파일 생성

    with open(csv_file, 'r') as infile, open(temp_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # 처음 두 줄 스킵
        next(reader)
        next(reader)

        for row in reader:
            writer.writerow(row)

    # 기존 파일 삭제 및 임시 파일 이름 변경
    os.remove(csv_file)
    os.rename(temp_file, csv_file)

def get_file_names(folder_path):
    file_names = []
    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            file_names.append(file_name)
    return file_names

folder_path = "./EDA"

file_names = get_file_names(folder_path)
print(file_names)
script_path = os.path.abspath(__file__)
current_directory = os.path.dirname(script_path)
file_names = get_file_names(current_directory)
print(file_names)
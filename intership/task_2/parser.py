import os
import shutil
import json
import random

def list_and_copy_photo_files(source_directory, target_directory, start_id, class_id):
    try:
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        files = os.listdir(source_directory)
        photo_files = [file for file in files if file.startswith('photo')]
        result = []
        if photo_files:
            for file in photo_files:
                result.append({"id": start_id, "name": file, "class": class_id})
                start_id += 1
                source_file = os.path.join(source_directory, file)
                target_file = os.path.join(target_directory, file)
                shutil.copy(source_file, target_file)
        else:
            print("Файлы, начинающиеся с 'photo', не найдены.")

    except FileNotFoundError:
        print(f"Папка '{source_directory}' не найдена.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
    finally:
        return start_id, result

def split_train_test(data, test_ratio=0.2):
    random.shuffle(data)
    split_index = int(len(data) * (1 - test_ratio))
    train_data = data[:split_index]
    test_data = data[split_index:]
    return train_data, test_data

def main():
    source_directory_paths = {0: "cats_files", 1: "cheetah_files", 2: "dogs_files", 3: "lion_files", 4: "monkey_files"}
    target_directory_path = "data"
    start_id = 0
    data = []
    for i in source_directory_paths.keys():
        start_id, result = list_and_copy_photo_files(source_directory_paths[i], target_directory_path, start_id, i)
        data.extend(result)

    with open("images.json", "w") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    train_data, test_data = split_train_test(data, test_ratio=0.2)

    train_ids = [item["id"] for item in train_data]
    test_ids = [item["id"] for item in test_data]

    with open("train.json", "w") as train_file:
        json.dump(train_ids, train_file, indent=4, ensure_ascii=False)

    with open("validation.json", "w") as test_file:
        json.dump(test_ids, test_file, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()

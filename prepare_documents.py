from datasets import load_dataset
import os
import json
from datasets.utils.logging import set_verbosity_error
import tqdm

# Tắt các thông báo verbose của datasets
set_verbosity_error()

# Định nghĩa các thư mục
CACHE_DIR = "./cache"
DOCUMENT_DIR = "./documents"

# Danh sách các dataset IDs
dataset_ids = [
    "burgerbee/pedagogy_textbook",
    "burgerbee/pedagogy_wiki",
    "burgerbee/psychology_textbook",
    "burgerbee/psychology_wiki",
    "burgerbee/psychiatry_textbook",
    "burgerbee/psychiatry_wiki",
    "burgerbee/art_and_culture_textbook",
    "burgerbee/art_and_culture_wiki",
    "burgerbee/medicine_textbook",
    "burgerbee/medicine_wiki",
    "burgerbee/chemistry_textbook",
    "burgerbee/chemistry_wiki",
    "burgerbee/social_studies_textbook",
    "burgerbee/social_studies_wiki",
    "burgerbee/religion_textbook",
    "burgerbee/religion_wiki",
    "burgerbee/science_studies_textbook",
    "burgerbee/science_studies_wiki",
    "burgerbee/history_textbook",
    "burgerbee/history_wiki",
    "burgerbee/philosophy_textbook",
    "burgerbee/philosophy_wiki",
    "burgerbee/biology_textbook",
    "burgerbee/biology_wiki",
    "burgerbee/physics_textbook",
    "burgerbee/physics_wiki",
]

# Tạo thư mục documents nếu chưa tồn tại
os.makedirs(DOCUMENT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Tải xuống và xử lý từng dataset
for dataset_id in tqdm.tqdm(dataset_ids, desc="Downloading datasets"):
    try:
        print(f"Downloading {dataset_id}...")
        dataset = load_dataset(dataset_id, cache_dir=CACHE_DIR)
        print(f"Downloaded {dataset_id} with {len(dataset)} splits.")
        
        # Tạo thư mục cho dataset
        dataset_name = dataset_id.split("/")[1]  # Lấy phần sau dấu "/"
        dataset_dir = os.path.join(DOCUMENT_DIR, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Xử lý từng split trong dataset
        for split in dataset:
            print(f"  - {split}: {len(dataset[split])} examples")
            split_data = len(dataset[split])
            
            # Xử lý từng sample trong split
            for i, sample in enumerate(dataset[split]):
                # Lấy thông tin từ sample
                title = sample["title"]
                text = sample["text"]
                
                # Thêm thông tin metadata
                sample["source"] = f"https://hf.co/datasets/{dataset_id}"
                sample["split"] = split
                sample["id"] = f"{dataset_id}-{split}-{i}"
                
                # Tạo tên file (thay thế ký tự "/" trong title để tránh lỗi)
                safe_title = title.replace('/', '_').replace('\\', '_')
                doc_filename = f"{safe_title}_{i}.json"
                doc_path = os.path.join(dataset_dir, doc_filename)
                
                # Lưu sample vào file JSON
                with open(doc_path, "w", encoding="utf-8") as f:
                    json.dump(sample, f, ensure_ascii=False, indent=4)
            
            print(f"Saved {split_data} samples to {dataset_dir}")
        
        print(f"Finished processing {dataset_id}.\n")
        
    except Exception as e:
        print(f"Error processing {dataset_id}: {str(e)}")
        continue

print("All datasets processed successfully.")
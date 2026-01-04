import os
from transformers import AutoTokenizer, AutoModel, AutoConfig

# Model siêu nhẹ để test
model_name = "prajjwal1/bert-tiny"
save_directory = "./models/bert-tiny"

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

print(f"--- Đang tải model SIÊU NHẸ: {model_name} ---")

try:
    # Tải Config
    config = AutoConfig.from_pretrained(model_name)
    config.save_pretrained(save_directory)

    # Tải Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_directory)

    # Tải Model
    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(save_directory)

    print(f"--- XONG! Model đã sẵn sàng tại: {save_directory} ---")
    print(f"Dung lượng thư mục này chỉ tầm 17-20MB thôi.")

except Exception as e:
    print(f"Vẫn gặp lỗi: {e}")
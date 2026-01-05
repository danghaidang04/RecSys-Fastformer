import pickle

cache_path = './data/speedy_data/train/bert-tiny_title+abstract_preprocessed_docs.pkl'

with open(cache_path, 'rb') as f:
    news_cache = pickle.load(f)
    
print(f"--- Kiểu dữ liệu của file cache: {type(news_cache)} ---")
# In ra các thuộc tính bên trong đối tượng này
print(f"--- Các thuộc tính tìm thấy: {dir(news_cache)} ---")

# Thông thường, MIND project sẽ có thuộc tính 'news_index' hoặc 'id2idx'
# Hãy thử tìm thuộc tính chứa dictionary
for attr in ['news_index', 'id2idx', 'newsid2idx', 'nid2idx']:
    if hasattr(news_cache, attr):
        data_dict = getattr(news_cache, attr)
        print(f"\n--- Tìm thấy thuộc tính: {attr} ---")
        print(f"5 ID đầu tiên: {list(data_dict.keys())[:5]}")
        break
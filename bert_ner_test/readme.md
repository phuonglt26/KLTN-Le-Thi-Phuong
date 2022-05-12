# Fine Tune PhoBert For NER Task

## Nội Dung
1. [Cài đặt](#setup) <br>
    [Tải model đã train về](#download_model)
2. [Đào tạo model](#train_model) <br>
    2.1 [Dữ liệu](#train_data) <br>
    2.2 [Đào tạo model](#train_model_script) <br>
3. [Đánh giá model](#evaluate_model) <br>
4. [Chạy suy luận](#inference) <br>

## 1. Cài đặt <a name="setup"></a>
```bash
pip install -r requirements.txt 
pip install -e .
```

## 2. Đào tạo model <a name="train_model"></a>
### 2.1 Dữ liệu <a name="train_data"></a>

### 2.2 Đào tạo model <a name="train_model_script"></a>
```bash
chmod +x run_train.sh
./run_train.sh
```
**Ý nghĩa các thông số:**
```
model_name_or_path: Đường dẫn đến file hoặc tên của model bert, ở đây dùng phobert thì giá trị đặt là 'vinai/phobert-base'
data_train: Đường dẫn đến file chứa dữ liệu train
output_dir: Đường dẫn đến file lưu model sau khi train
do_train: Thực hiện train nếu có tham số này
logging_steps: Trong quá trình training, sau logging_steps sẽ log ra loss và learning rate
save_steps: Model sẽ lưu checkpoints sau save_steps bước
learning_rate: Learning rate
max_seq_length: Chiều dài tối đa của 1 câu, câu ngắn hơn thì sẽ thêm <pad> để cho đủ max_seq_length, câu dài hơn max_seq_length sẽ bị cắt bớt
per_device_eval_batch_size: Batch size khi chạy evaluation
per_device_train_batch_size: Batch size khi train
num_train_epochs: Số Epochs train model
```

## 3. Đánh giá model <a name="evaluate_model"></a>

```python
from bert_entity_extractor import BertEntityExtractor

model = BertEntityExtractor("mounts/models/bert_ner_20_epochs/")
model.evaluate("mounts/data_vlsp/test.txt", batch_size=32, max_seq_length=80)
```


## 4. Chạy suy luận <a name="inference"></a>
Chạy suy luận:

```python
from bert_entity_extractor import BertEntityExtractor

model = BertEntityExtractor("mounts/models/bert_ner_20_epochs/")
model.predict([
    "Những khách hàng đầu tiên mua xe Mazda CX-8 2019 được áp dụng mức ưu đãi từ 1,149-1,399 tỷ đồng, sau đó xe sẽ về mức giá thực dao động từ 1,199-1,444 tỷ đồng, với mức giá trên đây là mẫu xe có giá cao nhất phân khúc SUV 7 chỗ tại Việt Nam"
])
```
Outputs:
```json

[
    {
        "text": "Những khách_hàng đầu_tiên mua xe Mazda CX - 8 2019 được áp_dụng mức ưu_đãi từ 1,149 - 1,399 tỷ đồng , sau đó xe sẽ về mức giá thực dao_động từ 1,199 - 1,444 tỷ đồng , với mức giá trên đây là mẫu xe có_giá cao nhất phân khúc SUV 7 chỗ tại Việt_Nam",
        "entities": [
            {"start": 5,
            "end": 10,
            "value": "Mazda CX - 8 2019",
            "entity": "SUB"},
            {"start": 48,
            "end": 55,
            "value": "phân khúc SUV 7 chỗ tại Việt_Nam",
            "entity": "OBJ"}
        ]
    }
]
```
import os
import numpy as np
import tensorflow as tf
from seqeval.metrics import classification_report
from typing import Dict, List, Tuple
from transformers import (
    AutoConfig,
    AutoTokenizer,
)
from bert_entity_extractor.utils_ner import TFTokenClassificationDataset
from bert_entity_extractor.models import TFRobertaForTokenClassification
from bert_entity_extractor.tasks import SubjectObjectDetect
from bert_entity_extractor.utils import normalize
from pyvi import ViTokenizer
from datetime import datetime


def get_root_path():
    root_path = os.path.dirname(os.path.realpath(__file__))
    return root_path


class BertEntityExtractor:
    def __init__(self, models_path="mounts/phobert_ner_model"):
        self.models_path = models_path
        self.tokenize_func = lambda text: ViTokenizer.tokenize(normalize(text))

        self.load_model()

    def load_model(self):
        self.token_classification_task = SubjectObjectDetect()
        self.labels = self.token_classification_task.get_labels()
        self.id2label = self.token_classification_task.id2label
        self.label2id = self.token_classification_task.label2id
        num_labels = len(self.labels)
        # Config Object for Model
        self.config = AutoConfig.from_pretrained(
            self.models_path,
            num_labels=num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )

        # Tokenizer Object
        self.tokenizer = AutoTokenizer.from_pretrained(self.models_path, use_fast=None)
        self.id2word = {value: key for key, value in self.tokenizer.get_vocab().items()}

        self.model = TFRobertaForTokenClassification.from_pretrained(
            self.models_path,
            from_pt=bool(".bin" in self.models_path),
            config=self.config,
        )
        return

    def predict(self, batch):
        batch = [self.tokenize_func(text) for text in batch]
        tokens = self.tokenizer(batch, padding=True, return_tensors="tf")
        y_preds = self.model(
            inputs=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            token_type_ids=tokens["token_type_ids"],
            training=False,
        )[0]
        y_preds = tf.math.argmax(y_preds, axis=-1)
        results = self.align_result(y_preds.numpy(), tokens["input_ids"].numpy(), batch)
        return self.convert_result(results)

    def labeling_data(self, file_in, file_out, batch_size=256):
        with open(file_in, "r") as file:
            data = file.read().splitlines()
        print(f"{datetime.now()}: Có tất cả {len(data)} câu")
        id2label_vectorize = np.vectorize(lambda ids: self.id2label[ids])
        start = 0
        labeled_data = []
        i = 0
        while start < len(data):
            end = min(start + batch_size, len(data))
            batch = [self.tokenize_func(text) for text in data[start:end]]
            tokens = self.tokenizer(batch, padding=True, return_tensors="tf")
            y_preds = self.model(
                inputs=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                token_type_ids=tokens["token_type_ids"],
                training=False,
            )[0]
            y_preds = tf.math.argmax(y_preds, axis=-1)
            y_preds = y_preds.numpy()[:, 1:-1]
            y_preds = id2label_vectorize(y_preds)
            labeled_data += [
                "\n".join([f"{x}\t{y}" for x, y in zip(doc.split(), label)])
                for doc, label in zip(batch, y_preds)
            ]
            start = end
            i += 1
            if i % 10 == 0:
                print(f"{datetime.now()}: Đã gán nhãn được cho {start} câu")
        with open(file_out, "w") as file:
            file.write("\n\n".join(labeled_data))
        print(f"{datetime.now()}: Gán nhãn xong.")
        print(f"{datetime.now()}: Dữ liệu gán nhãn đã được lưu ở {file_out}.")

    def evaluate(
        self,
        test_data_file,
        batch_size=128,
        max_seq_length=256,
    ):
        test_dataset = TFTokenClassificationDataset(
            token_classification_task=self.token_classification_task,
            data_dir=test_data_file,
            tokenizer=self.tokenizer,
            labels=self.labels,
            model_type=self.config.model_type,
            max_seq_length=max_seq_length,
            mode="",
        )
        test_dataset = test_dataset.get_dataset()
        test_dataset = (
            test_dataset.cache()
            .batch(batch_size=batch_size, drop_remainder=False)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        y_preds = None
        y_trues = None
        for x, y in test_dataset:
            pred = self.model(
                inputs=x["input_ids"],
                attention_mask=x["attention_mask"],
                token_type_ids=x["token_type_ids"],
                training=False,
            )[0]
            if y_preds is None:
                y_preds = tf.math.argmax(pred, axis=-1)
            else:
                y_preds = tf.concat((y_preds, tf.math.argmax(pred, axis=-1)), 0)

            if y_trues is None:
                y_trues = y
            else:
                y_trues = tf.concat((y_trues, y), 0)

        y_preds, y_trues = self.align_predictions(y_preds.numpy(), y_trues.numpy())
        report = classification_report(y_trues, y_preds, digits=4)
        print(report)

        return

    def align_result(self, y_preds, tokens, batch):
        results = []
        special_id = self.tokenizer.sep_token_id
        for t, sent_ids in enumerate(tokens):
            label_pred = y_preds[t]
            for index, idx in enumerate(sent_ids):
                if idx == special_id:
                    break

            sentences = map(lambda w_id: self.id2word[w_id], sent_ids[1:index])

            aligns = []
            flag_align = False
            for i, word in enumerate(sentences):
                if flag_align:
                    aligns[-1].append(i)
                else:
                    aligns.append([i])
                if len(word) > 2 and word[-2:] == "@@":
                    flag_align = True
                else:
                    flag_align = False
            labels = [self.id2label[label_pred[l[0] + 1]] for l in aligns]
            sentences = batch[t].split()
            assert len(sentences) == len(labels)
            results.append(list(zip(sentences, labels)))
        return results

    @staticmethod
    def convert_result(results):
        results2 = []
        for sentences in results:
            tmp = {"text": " ".join(map(lambda l: l[0], sentences)), "entities": []}

            value = []
            index = []
            for i, (word, tag) in enumerate(sentences):
                if tag != "O":
                    if tag.split("-")[0] == "B":
                        if value != []:
                            tmp["entities"].append(
                                {
                                    "start": index[0],
                                    "end": index[1],
                                    "value": " ".join(value),
                                    "entity": entity,
                                }
                            )
                        value = [word]
                        index = [i, i + 1]
                        entity = tag.split("-")[1]
                    else:
                        if value == []:
                            # Đây là nhưỡng trường hợp bị lỗi, không có B-ner ở trước I-ner
                            continue
                        value.append(word)
                        index[1] = i + 1
            if value != []:
                tmp["entities"].append(
                    {
                        "start": index[0],
                        "end": index[1],
                        "value": " ".join(value),
                        "entity": entity,
                    }
                )
            results2.append(tmp)
        return results2

    def align_predictions(
        self, preds: np.ndarray, label_ids: np.ndarray
    ) -> Tuple[List[int], List[int]]:
        batch_size, seq_len = preds.shape
        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != -100:
                    out_label_list[i].append(self.id2label[label_ids[i][j]])
                    preds_list[i].append(self.id2label[preds[i][j]])

        return preds_list, out_label_list

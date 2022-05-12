import logging
from typing import List, Union
from bert_entity_extractor.utils_ner import InputExample, Split, TokenClassificationTask


logger = logging.getLogger(__name__)


class SubjectObjectDetect(TokenClassificationTask):
    def __init__(self):
        self._map_labels = {
            0: "O",
            1: "B-SUB",
            2: "I-SUB",
            3: "B-OBJ",
            4: "I-OBJ",
        }

    def get_labels(self):
        return list(self._map_labels.values())

    @property
    def label2id(self):
        return {label: idx for idx, label in self._map_labels.items()}

    @property
    def id2label(self):
        return self._map_labels

    def read_examples_from_file(self, data_dir, mode="train"):
        with open(data_dir, "r") as file:
            data = file.read().splitlines()
        data = [
            self.line2sample(line.strip(), idx=i, mode=mode)
            for i, line in enumerate(data)
            if line.strip() != ""
        ]
        return data

    def map_labels(self, idx):
        try:
            return self._map_labels[int(idx)]
        except:
            raise ValueError(f"{idx} not in {self._map_labels.keys()}")

    def line2sample(self, line, idx=0, mode="train"):
        line = line.replace("\ufeff", "").split("\t")
        num_word = int(line[0])
        words = line[1: num_word+1]
        labels = line[num_word+1:]
        labels = [self.map_labels(lb) for lb in labels]
        sample = InputExample(guid=f"{mode}-{idx}", words=words, labels=labels)
        return sample

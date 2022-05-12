""" Fine-tuning the library models for named entity recognition."""

import logging
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoConfig,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    TFTrainer,
    TFTrainingArguments,
)
from transformers.utils import logging as hf_logging
from bert_entity_extractor.utils_ner import Split, TFTokenClassificationDataset

# custom model NER
from bert_entity_extractor.models import TFRobertaForTokenClassification
from bert_entity_extractor.tasks import SubjectObjectDetect


hf_logging.set_verbosity_info()
hf_logging.enable_default_handler()
hf_logging.enable_explicit_format()


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    use_fast: bool = field(
        default=False, metadata={"help": "Set this flag to use fast tokenization."}
    )
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_train: str = field(
        metadata={
            "help": "training data txt file"
        }
    )
    data_eval: Optional[str] = field(
        default=None,
        metadata={
            "help": "eval data txt file"
        }
    )
    max_seq_length: int = field(
        default=256,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TFTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    token_classification_task = SubjectObjectDetect()
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(
        "n_replicas: %s, distributed training: %s, 16-bits training: %s",
        training_args.n_replicas,
        bool(training_args.n_replicas > 1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Prepare Token Classification task
    labels = token_classification_task.get_labels()

    label2id = token_classification_task.label2id
    id2label = token_classification_task.id2label
    num_labels = len(labels)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
    )

    with training_args.strategy.scope():
        model = TFRobertaForTokenClassification.from_pretrained(
            model_args.model_name_or_path,
            from_pt=bool(".bin" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )

    # Get datasets
    train_dataset = (
        TFTokenClassificationDataset(
            token_classification_task=token_classification_task,
            data_dir=data_args.data_train,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.train,
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        TFTokenClassificationDataset(
            token_classification_task=token_classification_task,
            data_dir=data_args.data_eval,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.dev,
        )
        if training_args.do_eval
        else None
    )

    def align_predictions(
        predictions: np.ndarray, label_ids: np.ndarray
    ) -> Tuple[List[int], List[int]]:
        preds = np.argmax(predictions, axis=2)
        batch_size, seq_len = preds.shape
        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != -100:
                    out_label_list[i].append(id2label[label_ids[i][j]])
                    preds_list[i].append(id2label[preds[i][j]])

        return preds_list, out_label_list

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)

        return {
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
        }

    # Initialize our Trainer
    trainer = TFTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset.get_dataset() if train_dataset else None,
        eval_dataset=eval_dataset.get_dataset() if eval_dataset else None,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        result = trainer.evaluate()
        output_eval_file = os.path.join(data_args.result_dir, "eval_results.txt")

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key, value in result.items():
                print(f"{key:20} {value:10}")
                writer.write("\n")
                writer.write(f"{key:20}: {value:10}")

    # Predict
    if training_args.do_predict:
        logger.info("*** Test ***")
        test_dataset = TFTokenClassificationDataset(
            token_classification_task=token_classification_task,
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.test,
        )

        predictions, label_ids, _ = trainer.predict(test_dataset.get_dataset())
        preds_list, labels_list = align_predictions(predictions, label_ids)
        report = classification_report(labels_list, preds_list, digits=4)

        logger.info("***** Test results *****")
        logger.info("\n%s", report)

        output_test_results_file = os.path.join(
            data_args.result_dir, "test_results.txt"
        )

        with open(output_test_results_file, "w") as writer:
            writer.write("%s\n" % report)

        # Save predictions
        output_test_predictions_file = os.path.join(
            data_args.result_dir, "test_predictions.txt"
        )

        with open(output_test_predictions_file, "w") as writer:
            with open(os.path.join(data_args.data_dir, "test.txt"), "r") as f:
                example_id = 0

                for line in f:
                    if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                        writer.write(line)

                        if not preds_list[example_id]:
                            example_id += 1
                    elif preds_list[example_id]:
                        output_line = (
                            line.split("\t")[0]
                            + "\t"
                            + preds_list[example_id].pop(0)
                            + "\n"
                        )

                        writer.write(output_line)
                    else:
                        logger.warning(
                            "Maximum sequence length exceeded: No prediction for '%s'.",
                            line.split("\t")[0],
                        )

    return


if __name__ == "__main__":
    main()

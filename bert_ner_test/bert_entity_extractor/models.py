from transformers.modeling_tf_outputs import TFTokenClassifierOutput
from transformers.tokenization_utils_base import BatchEncoding
from transformers.modeling_tf_utils import get_initializer, shape_list
from transformers.modeling_tf_roberta import (
    TFRobertaPreTrainedModel,
    TFRobertaMainLayer,
)
import tensorflow as tf
import warnings


class TFTokenClassificationLoss:
    """
    Loss function suitable for token classification.

    .. note::

        Any label of -100 will be ignored (along with the corresponding logits) in the loss computation.

    """

    def compute_loss(self, labels, logits):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        # make sure only labels that are not equal to -100
        # are taken into account as loss
        if tf.math.reduce_any(labels == -1):
            warnings.warn(
                "Using `-1` to mask the loss for the token is deprecated. Please use `-100` instead."
            )
            active_loss = tf.reshape(labels, (-1,)) != -1
        else:
            active_loss = tf.reshape(labels, (-1,)) != -100
        reduced_logits = tf.boolean_mask(
            tf.reshape(logits, (-1, shape_list(logits)[2])), active_loss
        )
        labels = tf.boolean_mask(tf.reshape(labels, (-1,)), active_loss)

        return loss_fn(labels, reduced_logits)


class TFRobertaForTokenClassification(
    TFRobertaPreTrainedModel, TFTokenClassificationLoss
):

    authorized_missing_keys = [r"pooler"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.roberta = TFRobertaMainLayer(config, name="roberta")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(
            config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="classifier",
        )

    def call(
        self,
        inputs=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        training=False,
    ):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = (
            return_dict if return_dict is not None else self.roberta.return_dict
        )
        if isinstance(inputs, (tuple, list)):
            labels = inputs[9] if len(inputs) > 9 else labels
            if len(inputs) > 9:
                inputs = inputs[:9]
        elif isinstance(inputs, (dict, BatchEncoding)):
            labels = inputs.pop("labels", labels)

        outputs = self.roberta(
            inputs,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output, training=training)
        logits = self.classifier(sequence_output)

        loss = None if labels is None else self.compute_loss(labels, logits)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

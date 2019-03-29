# Copyright 2017, 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Functions to generate loss symbols for sequence-to-sequence models.
"""
import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import mxnet as mx
from mxnet.metric import EvalMetric

from . import config
from . import constants as C

logger = logging.getLogger(__name__)


class LossConfig(config.Config):
    """
    Loss configuration.

    :param name: Loss name.
    :param vocab_size: Target vocab size.
    :param normalization_type: How to normalize the loss.
    :param label_smoothing: Optional smoothing constant for label smoothing.
    """

    def __init__(self,
                 name: str,
                 vocab_size: int,
                 normalization_type: str,
                 label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.name = name
        self.vocab_size = vocab_size
        self.normalization_type = normalization_type
        self.label_smoothing = label_smoothing


def get_loss(loss_config: LossConfig) -> 'Loss':
    """
    Returns Loss instance.

    :param loss_config: Loss configuration.
    """
    if loss_config.name == C.CROSS_ENTROPY:
        return CrossEntropyLoss(loss_config)
    elif loss_config.name == C.CUSTOM_CROSS_ENTROPY:
        return CustomCrossEntropyLoss(loss_config)
    else:
        raise ValueError("unknown loss name: %s" % loss_config.name)


class Loss(ABC):
    """
    Generic Loss interface.
    get_loss() method should return a loss symbol and the softmax outputs.
    The softmax outputs (named C.SOFTMAX_NAME) are used by EvalMetrics to compute various metrics,
    e.g. perplexity, accuracy. In the special case of cross_entropy, the SoftmaxOutput symbol
    provides softmax outputs for forward() AND cross_entropy gradients for backward().
    """

    def get_loss(self,
                 logits: mx.sym.Symbol,
                 labels: mx.sym.Symbol,
                 weights: mx.sym.Symbol = None,
                 ignore_labels: List[int] = [C.PAD_ID],
                 grad_scale: float = 1.0,
                 prefix: str = '') -> List[mx.sym.Symbol]:
        """
        Returns loss and softmax output symbols given logits and integer-coded labels.

        :param preds: Shape: (batch_size * target_seq_len, target_vocab_size).
        :param labels: Shape: (batch_size * target_seq_len,).
        :param weights: Shape: (target_vocab_size).
        :param ignore_labels: List of label ids to be ignored.
        :param grad_scale: Scale the gradient by a float factor.
        :param prefix: Name prefix for the output.
        :return: List of loss symbol.
        """
        raise NotImplementedError()

    @abstractmethod
    def create_metric(self) -> EvalMetric:
        """
        Create an instance of the EvalMetric that corresponds to this Loss function.
        """
        pass


class CrossEntropyLoss(Loss):
    """
    Computes the cross-entropy loss.

    :param loss_config: Loss configuration.
    """

    def __init__(self, loss_config: LossConfig) -> None:
        logger.info("Loss: CrossEntropy(normalization_type=%s, label_smoothing=%s)",
                    loss_config.normalization_type, loss_config.label_smoothing)
        self.loss_config = loss_config

    def get_loss(self,
                 logits: mx.sym.Symbol,
                 labels: mx.sym.Symbol,
                 weights: mx.sym.Symbol = None,
                 ignore_labels: List[int] = [C.PAD_ID],
                 grad_scale: float = 1.0,
                 prefix: str = '') -> List[mx.sym.Symbol]:
        """
        Returns loss and softmax output symbols given logits and integer-coded labels.

        :param preds: Shape: (batch_size * target_seq_len, target_vocab_size).
        :param labels: Shape: (batch_size * target_seq_len,).
        :param weights: Not used for the loss, you may use CustomCrossEntropyLoss instead.
        :param ignore_labels: List of label ids to be ignored.
        :param grad_scale: Scale the gradient by a float factor.
        :param prefix: Name prefix for the output.
        :return: List of loss symbol.
        """
        if self.loss_config.normalization_type == C.LOSS_NORM_VALID:
            normalization = "valid"
        elif self.loss_config.normalization_type == C.LOSS_NORM_BATCH:
            normalization = "null"
        else:
            raise ValueError("Unknown loss normalization type: %s" % self.loss_config.normalization_type)

        if weights is not None:
            raise ValueError("Label weights are not supported for cross-entropy loss yet.")

        if len(ignore_labels) != 1:
            raise ValueError("You may specify one and only one ignore label for cross-entropy loss.")

        return [mx.sym.SoftmaxOutput(data=logits,
                                     label=labels,
                                     grad_scale=grad_scale,
                                     ignore_label=ignore_labels[0],
                                     use_ignore=True,
                                     normalization=normalization,
                                     smooth_alpha=self.loss_config.label_smoothing,
                                     name=prefix + C.SOFTMAX_NAME)]

    def create_metric(self) -> "CrossEntropyMetric":
        return CrossEntropyMetric(self.loss_config)


class CrossEntropyMetric(EvalMetric):
    """
    Version of the cross entropy metric that ignores padding tokens.

    :param loss_config: The configuration used for the corresponding loss.
    :param name: Name of this metric instance for display.
    :param output_names: Name of predictions that should be used when updating with update_dict.
    :param label_names: Name of labels that should be used when updating with update_dict.
    """

    def __init__(self,
                 loss_config: LossConfig,
                 name: str = C.CROSS_ENTROPY,
                 output_names: Optional[List[str]] = None,
                 label_names: Optional[List[str]] = None) -> None:
        super().__init__(name, output_names=output_names, label_names=label_names)
        self.loss_config = loss_config

    @staticmethod
    def cross_entropy(logprob, label):
        ce = -mx.nd.pick(logprob, label)  # pylint: disable=invalid-unary-operand-type
        return ce

    @staticmethod
    def cross_entropy_smoothed(logprob, label, alpha, num_classes):
        ce = CrossEntropyMetric.cross_entropy(logprob, label)
        # gain for each incorrect class
        per_class_gain = alpha / (num_classes - 1)
        # discounted loss for correct class
        ce *= 1 - alpha - per_class_gain
        # add gain for incorrect classes to total cross-entropy
        ce -= mx.nd.sum(logprob * per_class_gain, axis=-1, keepdims=False)
        return ce

    def update(self, labels, preds):
        for label, pred in zip(labels, preds):
            batch_size = label.shape[0]
            label = label.as_in_context(pred.context).reshape((label.size,))

            logprob = mx.nd.log(mx.nd.maximum(1e-10, pred))

            # ce: (batch*time,)
            if self.loss_config.label_smoothing > 0.0:
                ce = self.cross_entropy_smoothed(logprob, label,
                                                 alpha=self.loss_config.label_smoothing,
                                                 num_classes=self.loss_config.vocab_size)
            else:
                ce = self.cross_entropy(logprob, label)

            # mask pad tokens
            valid = (label != C.PAD_ID).astype(dtype=pred.dtype)
            ce *= valid

            ce = mx.nd.sum(ce)
            if self.loss_config.normalization_type == C.LOSS_NORM_VALID:
                num_valid = mx.nd.sum(valid)
                ce /= num_valid
                self.num_inst += 1
            elif self.loss_config.normalization_type == C.LOSS_NORM_BATCH:
                # When not normalizing, we divide by the batch size (number of sequences)
                # NOTE: This is different from MXNet's metrics
                self.num_inst += batch_size

            self.sum_metric += ce.asscalar()


class CustomCrossEntropyLoss(Loss):
    """
    Computes the cross-entropy loss.

    :param loss_config: Loss configuration.
    """

    def __init__(self, loss_config: LossConfig) -> None:
        logger.info("Loss: CustomCrossEntropy(normalization_type=%s)",
                    loss_config.normalization_type)
        self.loss_config = loss_config
        self.prefix = ''
        self.output = None

    @staticmethod
    def cross_entropy(logprob, labels):
        ce = -mx.sym.pick(logprob, labels, axis=1)
        return ce

    @staticmethod
    def cross_entropy_smoothed(logprob, labels, alpha, num_classes):
        ce = CustomCrossEntropyLoss.cross_entropy(logprob, labels)
        # gain for each incorrect class
        per_class_gain = alpha / (num_classes - 1)
        # discounted loss for correct class
        ce = ce * (1 - alpha - per_class_gain)
        # add gain for incorrect classes to total cross-entropy
        ce = ce - mx.sym.sum(logprob * per_class_gain, axis=-1, keepdims=False)
        return ce

    def get_outputs(self) -> List[mx.sym.Symbol]:
        return [mx.sym.MakeLoss(data=self.output, grad_scale=0, name=self.prefix + C.SOFTMAX_NAME)]

    def get_loss(self,
                 preds: mx.sym.Symbol,
                 labels: mx.sym.Symbol,
                 weights: mx.sym.Symbol = None,
                 ignore_labels: List[int] = [C.PAD_ID],
                 grad_scale: float = 1.0,
                 prefix: str = '') -> List[mx.sym.Symbol]:
        """
        Returns loss and softmax output symbols given output probability and integer-coded labels.

        :param preds: Shape: (batch_size * target_seq_len, target_vocab_size).
        :param labels: Shape: (batch_size * target_seq_len,).
        :param weights: Shape: (target_vocab_size).
        :param ignore_labels: List of label ids to be ignored.
        :param grad_scale: Scale the gradient by a float factor.
        :param prefix: Name prefix for the output.
        :return: List of loss symbol.
        """
        self.prefix = prefix
        self.output = preds
        # logprob: (batch_size * target_seq_len, target_vocab_size)
        logprob = mx.sym.log(mx.sym.maximum(preds, 1e-10))

        if weights is not None:
            logprob = mx.sym.broadcast_mul(logprob, weights)

        # ce: (batch_size * target_seq_len,)
        if self.loss_config.label_smoothing > 0.0:
            ce = self.cross_entropy_smoothed(logprob, labels,
                                             alpha=self.loss_config.label_smoothing,
                                             num_classes=self.loss_config.vocab_size)
        else:
            ce = self.cross_entropy(logprob, labels)

        # ignore labels
        valid = mx.sym.ones_like(labels)
        for ignore_id in ignore_labels:
            valid = mx.sym.broadcast_logical_and(valid, labels != ignore_id)
        ce = ce * valid

        # normalization
        if self.loss_config.normalization_type == C.LOSS_NORM_VALID:
            ce = mx.sym.broadcast_div(ce, mx.sym.sum(valid, keepdims=True))

        return [mx.sym.MakeLoss(data=ce,
                                grad_scale=grad_scale,
                                name=prefix + C.CUSTOM_CROSS_ENTROPY)]

    def create_metric(self) -> "CrossEntropyMetric":
        return CrossEntropyMetric(self.loss_config)

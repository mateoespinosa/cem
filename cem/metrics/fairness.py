"""
Module containing a set of fairness metrics of interest for our evaluation.
"""

import numpy as np
import sklearn.metrics

def make_discrete(y_pred, threshold=0.5):
    """Discretizes an array representing probabilities for each possible
    class into an array containing the labels of the prediction made for
    each sample (rather than its probability).

    Args:
        y_pred (np.ndarray): Predicted probabilities for each of the samples. If
            the task is binary, then we expect a numpy array containing numbers
            in [0, 1] with shape [B, 1] or [B], where B is the number of samples
            in the batch. If the task is multi-class/categorical, then we
            expect y_pred to have shape [B, n_classes].
        threshold (float, optional): If task is a binary task, then this is
            the threshold to use for binarizing the probabilities in y_pred.
            Defaults to 0.5.

    Returns:
        np.ndarray: An array with shape [B] containing the class with the
            highest probability for each sample of y_pred.
    """
    if (len(y_pred.shape) < 2) or (y_pred.shape[-1] == 1):
        # Then simply compute the actual accuracy
        y_pred = (y_pred >= threshold).astype(np.int32)
    else:
        # Else this is the multi label case
        y_pred = np.argmax(y_pred, axis=-1)
    return y_pred


def worst_group_accuracy(
    y_true,
    y_pred,
    attributes,
    n_labels,
    threshold=0.5,
    attribute_idxs=None,
    check_subgroups=False,
):
    """Computes the worst group accuracy given prediction y_pred and ground
    truth labels y_true. We assume all groups are defined by the binary vector
    attributes.

    Args:
        y_true (np.ndarray): Target classes for each of the samples. We expect
            this to be an np.ndarray with shape [B], where B is the number of
            samples in the batch, such that each element is an integer in the
            set {0, 1, ..., n_classes - 1}.
        y_pred (np.ndarray): Predicted probabilities for each of the samples. If
            the task is binary, then we expect a numpy array containing numbers
            in [0, 1] with shape [B, 1] or [B], where B is the number of samples
            in the batch. If the task is multi-class/categorical, then we
            expect y_pred to have shape [B, n_classes].
        attributes (np.ndarray): A binary vector with shape [B, n_attributes]
            containing indicators as to which groups each of the samples belongs
            to. If check_subgroups is False, then we will use these groups to
            determine the wrost group accuracy.
        n_labels (int): Number of class labels one can possibly find in y_true.
        threshold (float, optional): If task is a binary task, then this is
            the threshold to use for binarizing the probabilities in y_pred.
            Defaults to 0.5.
        attribute_idxs (np.ndarray, optional): If provided, then this vector
            of indices defines which attributes in `attributes` we should use
            to determine the worst group accuracy. If None, then we will use all
            the attributes provided. If empty, then this corresponds to the
            average accuracy. Defaults to None.
        check_subgroups (bool, optional): If True, then we will compute the
            loss accross all subgroups defined by the cartesian product between
            labels and attributes. This means that every pair (label, group) is
            considered a subgroup. Defaults to False.

    Raises:
        ValueError: if a sample does not belong to any group as defined by the
            `attributes` array.

    Returns:
        float: the worst group accuracy
    """
    y_pred = make_discrete(y_pred, threshold=threshold)
    accs = []
    if attribute_idxs is None:
        attribute_idxs = list(range(attributes.shape[-1]))
    if (len(attribute_idxs) == 0) or (
        (attributes is None) or
        (len(attributes) == 0)
    ):
        # Then we simply return the average accuracy
        return sklearn.metrics.accuracy_score(
            y_true,
            y_pred,
        )
    if check_subgroups:
        for y_label in range(n_labels):
            for attribute_idx in attribute_idxs:
                for pos_val in [0, 1]:
                    selected = np.logical_and(
                        attributes[:, attribute_idx] == pos_val,
                        y_true == y_label,
                    )
                    if not np.any(selected):
                        continue
                    y_true_with_attribute = y_true[selected]
                    y_pred_with_attribute = y_pred[selected]
                    accs.append(sklearn.metrics.accuracy_score(
                        y_true_with_attribute,
                        y_pred_with_attribute,
                    ))
    else:
        for attribute_idx in attribute_idxs:
            for pos_val in [0, 1]:
                selected = attributes[:, attribute_idx] == pos_val
                if not np.any(selected):
                    continue
                y_true_with_attribute = y_true[selected]
                y_pred_with_attribute = y_pred[selected]
                accs.append(sklearn.metrics.accuracy_score(
                    y_true_with_attribute,
                    y_pred_with_attribute,
                ))
    if accs:
        return np.min(accs)
    # Else there is no groups we could find here!
    raise ValueError(
        'We could not find any groups in the given batch when computing '
        'the worst group accuracy!'
    )

def average_group_accuracy(
    y_true,
    y_pred,
    attributes,
    n_labels,
    threshold=0.5,
    attribute_idxs=None,
    check_subgroups=False,
):
    """Computes the average group accuracy given prediction y_pred and ground
    truth labels y_true. We assume all groups are defined by the binary vector
    attributes.

    Args:
        y_true (np.ndarray): Target classes for each of the samples. We expect
            this to be an np.ndarray with shape [B], where B is the number of
            samples in the batch, such that each element is an integer in the
            set {0, 1, ..., n_classes - 1}.
        y_pred (np.ndarray): Predicted probabilities for each of the samples. If
            the task is binary, then we expect a numpy array containing numbers
            in [0, 1] with shape [B, 1] or [B], where B is the number of samples
            in the batch. If the task is multi-class/categorical, then we
            expect y_pred to have shape [B, n_classes].
        attributes (np.ndarray): A binary vector with shape [B, n_attributes]
            containing indicators as to which groups each of the samples belongs
            to. If check_subgroups is False, then we will use these groups to
            determine the wrost group accuracy.
        n_labels (int): Number of class labels one can possibly find in y_true.
        threshold (float, optional): If task is a binary task, then this is
            the threshold to use for binarizing the probabilities in y_pred.
            Defaults to 0.5.
        attribute_idxs (np.ndarray, optional): If provided, then this vector
            of indices defines which attributes in `attributes` we should use
            to determine the average group accuracy. If None, then we will use all
            the attributes provided. If empty, then this corresponds to the
            average accuracy. Defaults to None.
        check_subgroups (bool, optional): If True, then we will compute the
            loss accross all subgroups defined by the cartesian product between
            labels and attributes. This means that every pair (label, group) is
            considered a subgroup. Defaults to False.

    Raises:
        ValueError: if a sample does not belong to any group as defined by the
            `attributes` array.

    Returns:
        float: the average group accuracy
    """
    y_pred = make_discrete(y_pred, threshold=threshold)
    accs = []
    if attribute_idxs is None:
        attribute_idxs = list(range(attributes.shape[-1]))
    if (len(attribute_idxs) == 0) or (
        (attributes is None) or
        (len(attributes) == 0)
    ):
        # Then we simply return the average accuracy
        return sklearn.metrics.accuracy_score(
            y_true,
            y_pred,
        )
    if check_subgroups:
        for y_label in range(n_labels):
            for attribute_idx in attribute_idxs:
                for pos_val in [0, 1]:
                    selected = np.logical_and(
                        attributes[:, attribute_idx] == pos_val,
                        y_true == y_label,
                    )
                    if not np.any(selected):
                        continue
                    y_true_with_attribute = y_true[selected]
                    y_pred_with_attribute = y_pred[selected]
                    accs.append(sklearn.metrics.accuracy_score(
                        y_true_with_attribute,
                        y_pred_with_attribute,
                    ))
    else:
        for attribute_idx in attribute_idxs:
            for pos_val in [0, 1]:
                selected = attributes[:, attribute_idx] == pos_val
                if not np.any(selected):
                    continue
                y_true_with_attribute = y_true[selected]
                y_pred_with_attribute = y_pred[selected]
                accs.append(sklearn.metrics.accuracy_score(
                    y_true_with_attribute,
                    y_pred_with_attribute,
                ))
    if accs:
        return np.mean(accs)
    # Else there is no groups we could find here!
    raise ValueError(
        'We could not find any groups in the given batch when computing '
        'the average group accuracy!'
    )


def worst_group_auc(
    y_true,
    y_pred,
    attributes,
    multi_class='ovo',
    attribute_idxs=None,
):
    """Computes the worst group AUC given prediction y_pred and ground
    truth labels y_true. We assume all groups are defined by the binary vector
    attributes.

    Args:
        y_true (np.ndarray): Target classes for each of the samples. We expect
            this to be an np.ndarray with shape [B], where B is the number of
            samples in the batch, such that each element is an integer in the
            set {0, 1, ..., n_classes - 1}.
        y_pred (np.ndarray): Predicted probabilities for each of the samples. If
            the task is binary, then we expect a numpy array containing numbers
            in [0, 1] with shape [B, 1] or [B], where B is the number of samples
            in the batch. If the task is multi-class/categorical, then we
            expect y_pred to have shape [B, n_classes].
        attributes (np.ndarray): A binary vector with shape [B, n_attributes]
            containing indicators as to which groups each of the samples belongs
            to.
        attribute_idxs (np.ndarray, optional): If provided, then this vector
            of indices defines which attributes in `attributes` we should use
            to determine the worst group AUC. If None, then we will use all
            the attributes provided. If empty, then this corresponds to the
            average AUC. Defaults to None.
        multi_class (str, optional): When dealing with multi-class target labels
            this argument indicates whether we should evaluate the one-vs-one
            AUC-ROC ('ovo') or the one-vs-all AUC-ROC ('ova') to produce a
            single scalar score for the AUC.

    Raises:
        ValueError: if a sample does not belong to any group as defined by the
            `attributes` array.

    Returns:
        float: the worst group AUC
    """
    aucs = []
    if attribute_idxs is None:
        attribute_idxs = list(range(attributes.shape[-1]))
    if (len(attribute_idxs) == 0) or (
        (attributes is None) or
        (len(attributes) == 0)
    ):
        # Then we simply return the average accuracy
        return sklearn.metrics.roc_auc_score(
            y_true,
            y_pred,
            multi_class=multi_class,
        )
    for attribute_idx in attribute_idxs:
        for pos_val in [0, 1]:
            selected = attributes[:, attribute_idx] == pos_val
            y_true_with_attribute = y_true[selected]
            y_pred_with_attribute = y_pred[selected]

            if (len(y_pred.shape) < 2) or (y_pred.shape[-1] == 1):
                # Then simply compute the actual accuracy
                aucs.append(sklearn.metrics.roc_auc_score(
                    y_true_with_attribute,
                    y_pred_with_attribute,
                ))

            # Else this is the multi label case
            aucs.append(sklearn.metrics.roc_auc_score(
                y_true_with_attribute,
                y_pred_with_attribute,
                multi_class=multi_class,
            ))

    if aucs:
        return np.min(aucs)
    # Else there is no groups we could find here!
    raise ValueError(
        'We could not find any groups in the given batch when computing '
        'the worst group AUC!'
    )

def fpr_difference(
    y_true,
    y_pred,
    attributes,
    attribute_idx,
    threshold=0.5,
    complement=False,
    positive_class=1,
):
    """Computes the difference in false positive rate (FPR) of predictions
    y_pred given ground truth y_true for samples that have attribute at index
    `attribute_idx` on vs samples that have attribute at index `attribute_idx`
    off. This method computes
        FPR(samples with attributes[attribute_idx] == 0) -
        FPR(samples with attributes[attribute_idx] == 1)
    if `complement` is False. Otherwise, it outputs the negation:
        FPR(samples with attributes[attribute_idx] == 1) -
        FPR(samples with attributes[attribute_idx] == 0)
    Args:
        y_true (np.ndarray): Target classes for each of the samples. We expect
            this to be an np.ndarray with shape [B], where B is the number of
            samples in the batch, such that each element is an integer in the
            set {0, 1, ..., n_classes - 1}.
        y_pred (np.ndarray): Predicted probabilities for each of the samples. If
            the task is binary, then we expect a numpy array containing numbers
            in [0, 1] with shape [B, 1] or [B], where B is the number of samples
            in the batch. If the task is multi-class/categorical, then we
            expect y_pred to have shape [B, n_classes].
        attributes (np.ndarray): A binary vector with shape [B, n_attributes]
            containing indicators as to which groups each of the samples belongs
            to.
        attribute_idx (int): attribute index which we will use in `attributes`
            to determine the two groups whose FPRs we will compare.
        threshold (float, optional): If task is a binary task, then this is
            the threshold to use for binarizing the probabilities in y_pred.
            Defaults to 0.5.
        complement (bool, optional): Whether we output FPR(attr == 0) -
            FPR(attr == 1) or the negation. Defaults to False which means we
            output FPR(attr == 0) - FPR(attr == 1).
        positive_class (int, optional): Value used to represent the ground truth
            positive class. Defaults to 1.


    Returns:
        float: fpr difference between the two groups
    """
    y_pred = make_discrete(y_pred, threshold=threshold)
    fprs = []
    if (attributes is None) or (len(attributes.shape) <= 1) or (
        (attributes.shape[-1] == 0)
    ):
        # Then we simply return 0
        return 0
    for attribute_val in [0 , 1]:
        selected = attributes[:, attribute_idx] == attribute_val
        y_true_with_attribute = (y_true[selected] == positive_class).astype(
            np.int32
        )
        y_true_with_attribute = np.reshape(y_true_with_attribute, -1)
        y_pred_with_attribute = (y_pred[selected] == positive_class).astype(
            np.int32
        )
        y_pred_with_attribute = np.reshape(y_pred_with_attribute, -1)

        cnf_matrix = sklearn.metrics.confusion_matrix(
            y_true_with_attribute,
            y_pred_with_attribute,
            labels=[0, 1],
        )

        tn, fp, fn, tp = cnf_matrix.ravel()
        fprs.append(fp/(fp + tn))
    if len(fprs) == 0:
        return 0
    if complement:
        return fprs[0] - fprs[1]
    return fprs[1] - fprs[0]
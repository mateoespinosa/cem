import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_selection import mutual_info_classif
from scipy.special import softmax
from collections import defaultdict
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def niche_completeness(c_pred, y_true, predictor_model, niches):
    '''
    Computes the niche completeness score for the downstream task
    :param c_pred: Concept data predictions, numpy array of shape
        (n_samples, n_concepts)
    :param y_true: Ground-truth task label data, numpy array of shape
        (n_samples, n_tasks)
    :param predictor_model: trained decoder model to use for predicting the task
        labels from the concept data
    :return: Accuracy of predictor_model, evaluated on niches obtained from the
        provided concept and label data
    '''
    n_tasks = y_true.shape[1]
    # compute niche completeness for each task
    niche_completeness_list, y_pred_list = [], []
    for task in range(n_tasks):
        # find niche
        niche = np.zeros_like(c_pred)
        niche[:, niches[:, task] > 0] = c_pred[:, niches[:, task] > 0]

        # compute task predictions
        y_pred_niche = predictor_model.predict_proba(niche)
        if predictor_model.__class__.__name__ == 'Sequential':
            # get class labels from logits
            y_pred_niche = y_pred_niche > 0
        elif len(y_pred_niche.shape) == 1:
            y_pred_niche = y_pred_niche[:, np.newaxis]

        y_pred_list.append(y_pred_niche[:, task])

    y_preds = np.vstack(y_pred_list).T
    y_preds = softmax(y_preds, axis=1)
    auc = roc_auc_score(y_true, y_preds, multi_class='ovo')

    result = {
        'auc_completeness': auc,
        'y_preds': y_preds,
    }
    return result


def niche_completeness_ratio(c_pred, y_true, predictor_model, niches):
    '''
    Computes the niche completeness ratio for the downstream task
    :param c_pred: Concept data predictions, numpy array of shape
        (n_samples, n_concepts)
    :param y_true: Ground-truth task label data, numpy array of shape
        (n_samples, n_tasks)
    :param predictor_model: sklearn model to use for predicting the task labels
        from the concept data
    :return: Accuracy ratio between the accuracy of predictor_model evaluated
        on niches and the accuracy of predictor_model evaluated on all concepts
    '''
    n_tasks = y_true.shape[1]

    y_pred_test = predictor_model.predict_proba(c_pred)
    if predictor_model.__class__.__name__ == 'Sequential':
        # get class labels from logits
        y_pred_test = y_pred_test > 0
    elif len(y_pred_test.shape) == 1:
        y_pred_test = y_pred_test[:, np.newaxis]

    # compute niche completeness for each task
    niche_completeness_list = []
    for task in range(n_tasks):
        # find niche
        niche = np.zeros_like(c_pred)
        niche[:, niches[:, task] > 0] = c_pred[:, niches[:, task] > 0]

        # compute task predictions
        y_pred_niche = predictor_model.predict_proba(niche)
        if predictor_model.__class__.__name__ == 'Sequential':
            # get class labels from logits
            y_pred_niche = y_pred_niche > 0
        elif len(y_pred_niche.shape) == 1:
            y_pred_niche = y_pred_niche[:, np.newaxis]

        # compute accuracies
        accuracy_base = accuracy_score(y_true[:, task], y_pred_test[:, task])
        accuracy_niche = accuracy_score(y_true[:, task], y_pred_niche[:, task])

        # compute the accuracy ratio of the niche w.r.t. the baseline
        # (full concept bottleneck) the higher the better (high predictive power
        # of the niche)
        niche_completeness = accuracy_niche / accuracy_base
        niche_completeness_list.append(niche_completeness)

    result = {
        'niche_completeness_ratio_mean': np.mean(niche_completeness_list),
        'niche_completeness_ratio': niche_completeness_list,
    }
    return result


def niche_impurity(c_pred, y_true, predictor_model, niches):
    '''
    Computes the niche impurity score for the downstream task
    :param c_pred: Concept data predictions, numpy array of shape
        (n_samples, n_concepts)
    :param y_true: Ground-truth task label data, numpy array of shape
        (n_samples, n_tasks)
    :param predictor_model: sklearn model to use for predicting the task labels
        from the concept data
    :return: Accuracy ratio between the accuracy of predictor_model evaluated on
        concepts outside niches and the accuracy of predictor_model evaluated on
        concepts inside niches
    '''
    n_tasks = y_true.shape[1]

    if len(c_pred.shape) == 2:
        n_samples, n_concepts = c_pred.shape
        assert n_concepts == n_tasks, 'Number of concepts and tasks must be equal'

        # compute niche completeness for each task
        y_pred_list = []

        nis = 0.0
        count = 0
        for i in range(n_concepts):
            if len(np.unique(y_true[:, i])) == 1:
                continue
            count += 1
            # find niche
            niche = np.zeros_like(c_pred)
            niche[:, niches[:, i] > 0] = c_pred[:, niches[:, i] > 0]

            # find concepts outside the niche
            niche_out = np.zeros_like(c_pred)
            niche_out[:, niches[:, i] <= 0] = c_pred[:, niches[:, i] <= 0]

            # compute task predictions
            y_pred_niche = predictor_model.predict_proba(niche)
            y_pred_niche_out = predictor_model.predict_proba(niche_out)
            if predictor_model.__class__.__name__ == 'Sequential':
                # get class labels from logits
                y_pred_niche_out = y_pred_niche_out > 0
            elif len(y_pred_niche.shape) == 1:
                y_pred_niche_out = y_pred_niche_out[:, np.newaxis]

            nis += roc_auc_score(y_true[:, i], y_pred_niche_out[:, i])
        if count:
            nis = nis / count
    else:
        n_samples, h_concepts, n_concepts = c_pred.shape
        assert n_concepts == n_tasks, 'Number of concepts and tasks must be equal'
        c_soft_test2 = c_pred.reshape(-1, h_concepts*n_concepts)
        nis = 0.0
        count = 0
        for i in range(n_concepts):
            if len(np.unique(y_true[:, i])) == 1:
                continue
            count += 1
            c_soft_test3 = c_soft_test2.copy()
            mask = np.repeat(niches[:, i], h_concepts)
            c_soft_test_masked = c_soft_test3
            c_soft_test_masked[:, mask] = 0
            c_pred_niche = predictor_model.predict_proba(c_soft_test_masked)[:, i]

            c_soft_test3 = c_soft_test2.copy()
            c_soft_test_masked = c_soft_test3
            c_soft_test_masked[:, ~mask] = 0
            c_pred_niche = predictor_model.predict_proba(c_soft_test_masked)[:, i]
            nis += roc_auc_score(
                y_true[:, i],
                c_pred_niche,
            )
        if count:
            nis = nis / count
    return nis


def niche_finding(c, y, mode='mi', threshold=0.5):
    n_concepts = c.shape[-1]
    n_targets = y.shape[-1]
    if len(c.shape) == 3:
        # Multi-dimensional concept representation case!
        n_samples, h_concepts, n_concepts = c.shape
        niching_matrix = np.zeros((n_concepts, n_targets))
        for j in range(n_targets):
            for i in range(n_concepts):
                corrm = np.corrcoef(np.hstack([c[:, :, i], y[:, j].reshape(-1, 1)]).T)
                nm = corrm[:h_concepts, h_concepts:]
                niching_matrix[i, j] = nm.max()
        niches = niching_matrix > threshold
    else:
        if mode == 'corr':
            corrm = np.corrcoef(np.hstack([c, y]).T)
            niching_matrix = corrm[:n_concepts, n_concepts:]
            niches = np.abs(niching_matrix) > threshold
        elif mode == 'mi':
            nm = []
            for yj in y.T:
                mi = mutual_info_classif(c, yj)
                nm.append(mi)
            nm = np.vstack(nm).T
            niching_matrix = nm / np.max(nm)
            niches = niching_matrix > threshold
        else:
            return None, None

    return niches, niching_matrix


def niching_high_dim(c_soft_train, c_true_train, c_soft_test, c_true_test, classifier, threshold=0.5):
    n_samples, h_concepts, n_concepts = c_soft_train.shape
    niching_matrix = np.zeros((n_concepts, n_concepts))
    for j in range(n_concepts):
        for i in range(n_concepts):
            corrm = np.corrcoef(np.hstack([c_soft_train[:, :, i], c_true_train[:, j].reshape(-1, 1)]).T)
            nm = corrm[:h_concepts, h_concepts:]
            niching_matrix[i, j] = nm.max()

    c_soft_train2 = c_soft_train.reshape(-1, h_concepts*n_concepts)
    c_soft_test2 = c_soft_test.reshape(-1, h_concepts*n_concepts)
    classifier.fit(c_soft_train2, c_true_train)

    c_preds_impurity = []
    niches = niching_matrix > threshold
    for i in range(n_concepts):
        c_soft_test3 = c_soft_test2.copy()
        mask = np.repeat(niches[:, i], h_concepts)
        c_soft_test_masked = c_soft_test3
        c_soft_test_masked[:, mask] = 0
        c_pred_niche = classifier.predict_proba(c_soft_test_masked)[:, i]

        c_soft_test3 = c_soft_test2.copy()
        c_soft_test_masked = c_soft_test3
        c_soft_test_masked[:, ~mask] = 0
        c_pred_niche = classifier.predict_proba(c_soft_test_masked)[:, i]
        c_preds_impurity.append(c_pred_niche)

    c_preds_impurity = np.stack(c_preds_impurity).T
    c_preds_impurity = softmax(c_preds_impurity, axis=1)
    return roc_auc_score(
        c_true_test.argmax(axis=1),
        c_preds_impurity,
        multi_class='ovo',
    )


def niche_impurity_score(
    c_soft,
    c_true,
    c_soft_train=None,
    c_true_train=None,
    predictor_model_fn=None,
    predictor_train_kwags=None,
    delta_beta=0.05,
    test_size=0.2,
):
    """
    Returns the niche impurity score (NIS) of the given soft concept
    representations `c_soft` with respect to their corresponding ground truth
    concepts `c_true`. This value is higher if concepts encode unnecessary
    information from other concepts distributed across SUBSETS of soft concept
    representations, and lower otherwise.

    :param Or[np.ndarray, List[np.ndarray]] c_soft: Predicted set of "soft"
        concept representations by a concept encoder model applied to the
        testing data. This argument must be an np.ndarray with shape
        (n_samples, ..., n_concepts) where the concept representation may be
        of any rank as long as the last dimension is the dimension used to
        separate distinct concept representations. If concepts have distinct
        array shapes for their representations, then this argument is expected
        to be a list of `n_concepts` np.ndarrays where the i-th element in the
        list is an array with shape (n_samples, ...) containing the tensor
        representation of the i-th concept.
        Note that in either case we only require that the first dimension.
    :param np.ndarray c_true: Ground truth concept values in one-to-one
        correspondence with concepts in c_soft. Shape must be
        (n_samples, n_concepts).
    :param Function[(int,), sklearn-like Estimator] predictor_model_fn: A
        function generator that takes as an argument the number of
        the output target concept and produces an sklearn-like Estimator
        which one can train for predicting a concept given another concept's
        soft concept values. If not given then we will use a 3-layer ReLU MLP
        as our predictor.
    :param Dict[Any, Any] predictor_train_kwags: optional arguments to pass
        the estimator being when calling its `fit` method.
    :param float test_size: A value in [0, 1] indicating the fraction of the
        given data that will be used to evaluate the trained concept-based
        classifier. The rest of the data will be used for training said
        classifier.

    :returns float: A non-negative float in [0, 1] representing the degree to
        which individual concepts in the given representations encode
        unnecessary information regarding other concepts distributed across
        them.
    """
    (n_samples, n_concepts) = c_true.shape
    # finding niches for several values of beta
    niche_impurities = []

    if predictor_model_fn is None:
        predictor_model_fn = lambda n_concepts: MLPClassifier(
            (20, 20),
            random_state=1,
            max_iter=1000,
            batch_size=min(512, n_samples)
        )
    if predictor_train_kwags is None:
        predictor_train_kwags = {}
    if len(c_soft.shape) == 2 and c_soft.shape[1] == 1:
        # Then get rid of degenerate dimension for simplicity
        c_soft = np.reshape(c_soft, (-1, n_concepts))
    if c_soft_train is not None and (
        len(c_soft_train.shape) == 2 and c_soft_train.shape[1] == 1
    ):
        # Then get rid of degenerate dimension for simplicity
        c_soft_train = np.reshape(c_soft_train, (-1, n_concepts))

    # And estimate the area under the curve using the trapezoid method
    auc = 0
    prev_value = None
    classifier = predictor_model_fn(n_concepts=n_concepts)
    if (c_soft_train is None) and (c_true_train is None):
        c_soft_train, c_soft_test, c_true_train, c_true_test = train_test_split(
            c_soft,
            c_true,
            test_size=test_size,
        )
    else:
        c_true_test = c_true
        c_soft_test = c_soft

    if len(c_soft_train.shape) == 3:
        # Then we are working in the multi-dimensional case!
        # So we need to flattent the dimensions
        classifier.fit(
            c_soft_train.reshape(c_soft_train.shape[0], -1),
            c_true_train,
            **predictor_train_kwags,
        )
    else:
        classifier.fit(c_soft_train, c_true_train, **predictor_train_kwags)

    for beta in tqdm(np.arange(0.0, 1.0, delta_beta)):
        niches, _ = niche_finding(
            c_soft_train,
            c_true_train,
            mode='corr',
            threshold=beta,
        )
        # compute impurity scores
        nis_score = niche_impurity(
            c_soft_test,
            c_true_test,
            classifier,
            niches,
        )
        niche_impurities.append(nis_score)
        # And update the area under the curve
        if prev_value is not None:
            auc += (prev_value + nis_score) * (delta_beta / 2)
        prev_value = nis_score

    return auc


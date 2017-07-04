import math
import statistics
import warnings
from collections import defaultdict

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


def _build_models(self, X=None, lengths=None, X_valid=None, lengths_valid=None):
    if X_valid is None:
        X_valid = self.X
    if lengths_valid is None:
        lengths_valid = self.lengths

    n_components = range(self.min_n_components, self.max_n_components + 1)
    for num_states in n_components:
        model = self.base_model(num_states, X, lengths)
        if model is None:
            continue
        try:
            score = model.score(X_valid, lengths_valid)
        except ValueError:
            score = np.inf
        yield num_states, model, score


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states, X=None, lengths=None):
        if X is None:
            X = self.X
        if lengths is None:
            lengths = self.lengths

        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        hmm_model = GaussianHMM(
            n_components=num_states, covariance_type="diag", n_iter=1000,
            random_state=self.random_state, verbose=False)

        try:
            hmm_model.fit(self.X, self.lengths)
        except ValueError:
            if self.verbose:
                print("failure on {} with {} states".format(
                    self.this_word, num_states))
        else:
            if self.verbose:
                print("model created for {} with {} states".format(
                    self.this_word, num_states))
            return hmm_model


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        n = len(self.X)
        models = []
        scores = []
        for num_states, model, logL in _build_models(self):
            bic = -2 * np.log(logL) + num_states * np.log(n)
            scores.append(bic)
            models.append(model)

        if models:
            return models[np.argmax(scores)]


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        hwords = self.hwords.copy()
        del hwords[self.this_word]
        n = len(hwords)

        best_model = None
        best_score = np.inf

        for num_states, model, logL in _build_models(self):
            other_scores = []
            for X, lengths in hwords.values():
                try:
                    other_logL = model.score(X, lengths)
                except ValueError:
                    pass
                else:
                    other_scores.append(other_logL)
            dic = logL - np.mean(other_scores)
            if dic < best_score:
                best_score = dic
                best_model = model

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        if len(self.lengths) < 3:
            return self.base_model(self.n_constant)

        best_score = np.inf
        best_num_states = self.n_constant

        lengths = np.array(self.lengths)
        idx = np.cumsum(lengths)
        idx = np.array([idx - lengths, idx]).T

        X = [self.X[i:j] for i, j in idx]

        scores = defaultdict(lambda: [])
        for cv_train_idx, cv_test_idx in KFold().split(self.lengths):
            X_train = np.concatenate([X[i] for i in cv_train_idx], axis=0)
            X_valid = np.concatenate([X[i] for i in cv_test_idx], axis=0)
            lengths_train = lengths[cv_train_idx]
            lengths_valid = lengths[cv_test_idx]
            models = _build_models(
                self, X=X_train, lengths=lengths_train,
                X_valid=X_valid, lengths_valid=lengths_valid)
            for num_states, _, score in models:
                scores[num_states].append(score)

        num_states = sorted(
            scores.items(), key=lambda it: np.mean(it[1]))[0][0]
        model = self.base_model(num_states)
        return model

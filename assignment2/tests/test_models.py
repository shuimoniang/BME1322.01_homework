import unittest

import numpy as np

from bayes_net import (
    FEATURE_CARDINALITIES,
    FEATURE_NAMES,
    FEATURE_PARENTS,
    BayesNetClassifier,
    extract_features,
)
from main import classification_metrics, confusion_matrix, stratified_split
from naive_bayes import NaiveBayesClassifier


def synthetic_mnist():
    images, labels = [], []
    for label in range(10):
        for variant in range(3):
            image = np.zeros((28, 28), dtype=np.uint8)
            column = 4 + 2 * label
            row = 3 + 2 * label
            image[:, column : min(column + 2 + variant, 28)] = 80 + label * 15
            image[row : min(row + 2, 28), :] = 255
            images.append(image.ravel())
            labels.append(label)
    return np.asarray(images), np.asarray(labels)


class ModelProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = synthetic_mnist()

    def test_invalid_alpha(self):
        for classifier in (NaiveBayesClassifier, BayesNetClassifier):
            with self.assertRaises(ValueError):
                classifier(alpha=0)
            with self.assertRaises(ValueError):
                classifier(alpha=np.nan)

    def test_unfitted_prediction(self):
        for classifier in (NaiveBayesClassifier, BayesNetClassifier):
            with self.assertRaises(RuntimeError):
                classifier().predict(self.X[:1])

    def test_input_validation(self):
        for classifier in (NaiveBayesClassifier, BayesNetClassifier):
            with self.assertRaises(ValueError):
                classifier().fit(self.X[:, :-1], self.y)
            invalid = self.X.astype(float)
            invalid[0, 0] = 256
            with self.assertRaises(ValueError):
                classifier().fit(invalid, self.y)

    def test_naive_bayes_probabilities(self):
        model = NaiveBayesClassifier(alpha=0.1).fit(self.X, self.y)
        probabilities = model.predict_proba(self.X[:7])
        self.assertEqual(probabilities.shape, (7, 10))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        self.assertEqual(model.predict(self.X[:7]).dtype, np.int64)
        np.testing.assert_allclose(
            np.exp(model.feature_log_prob_).sum(axis=-1), 1.0
        )

    def test_bayes_net_features_and_cpts(self):
        features = extract_features(self.X)
        self.assertEqual(features.shape, (len(self.X), 12))
        self.assertEqual(len(FEATURE_NAMES), 12)
        for index, cardinality in enumerate(FEATURE_CARDINALITIES):
            self.assertTrue(np.all((features[:, index] >= 0) & (features[:, index] < cardinality)))

        model = BayesNetClassifier(alpha=0.5).fit(self.X, self.y)
        probabilities = model.predict_proba(self.X[:7])
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        for cpt in model.cpt_log_prob_:
            np.testing.assert_allclose(np.exp(cpt).sum(axis=-1), 1.0)

    def test_dag_constraints(self):
        feature_edges = [(parent, child) for child, parent in enumerate(FEATURE_PARENTS) if parent >= 0]
        self.assertGreaterEqual(len(feature_edges), 2)
        self.assertTrue(all(parent < len(FEATURE_NAMES) for parent, _ in feature_edges))
        self.assertTrue(all(1 + int(parent >= 0) <= 2 for parent in FEATURE_PARENTS))

        children = {node: [] for node in range(len(FEATURE_NAMES))}
        indegree = np.zeros(len(FEATURE_NAMES), dtype=int)
        for parent, child in feature_edges:
            children[parent].append(child)
            indegree[child] += 1
        ready = [node for node, degree in enumerate(indegree) if degree == 0]
        visited = 0
        while ready:
            node = ready.pop()
            visited += 1
            for child in children[node]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        self.assertEqual(visited, len(FEATURE_NAMES))

    def test_empty_prediction(self):
        empty = np.empty((0, 784), dtype=np.uint8)
        for model in (
            NaiveBayesClassifier().fit(self.X, self.y),
            BayesNetClassifier().fit(self.X, self.y),
        ):
            self.assertEqual(model.predict(empty).shape, (0,))
            self.assertEqual(model.predict_proba(empty).shape, (0, 10))


class EvaluationTests(unittest.TestCase):
    def test_metrics(self):
        truth = np.arange(10)
        prediction = np.arange(10)
        matrix = confusion_matrix(truth, prediction)
        self.assertEqual(int(matrix.sum()), 10)
        metrics = classification_metrics(truth, prediction)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)

    def test_stratified_split_is_reproducible(self):
        y = np.repeat(np.arange(10), 12)
        first = stratified_split(y)
        second = stratified_split(y)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(len(first[1]), 20)


if __name__ == "__main__":
    unittest.main()


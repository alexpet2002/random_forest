# import numpy as np
# from collections import Counter
#
# class DecisionStump:
#     def __init__(self, matrix):
#         self.gini_index = 0
#         self.matrix = matrix
#         self.threshold = None
#         self.alpha = None
#         self.prediction = None
#
#     def calculate_weights(self, n):
#         initial_matrix = self.matrix
#         weights = 1/n
#         for element in initial_matrix:
#             element.append(weights)
#
#     def calculate_gini_index(self):
#         total_instances = len(self.matrix)
#
#         if total_instances == 0:
#             return 0
#
#         class_counts = Counter(instance[-1] for instance in self.matrix)
#         gini_index = 1.0
#
#         for class_label in class_counts:
#             probability = class_counts[class_label] / total_instances
#             gini_index -= probability ** 2
#
#         return gini_index
#
# def train_decision_stump(X, y, weights):
#     num_samples, num_features = X.shape
#     min_error = float('inf')
#
#     for feature_index in range(num_features):
#         unique_values = np.unique(X[:, feature_index])
#         for threshold in unique_values:
#             predictions = np.ones(num_samples)
#             predictions[X[:, feature_index] <= threshold] = -1
#             error = np.sum(weights * (predictions != y))
#
#             if error < min_error:
#                 min_error = error
#                 best_stump = DecisionStump()
#                 best_stump.feature_index = feature_index
#                 best_stump.threshold = threshold
#                 best_stump.prediction = predictions.copy()
#
#     return best_stump, min_error
#
#
# def update_weights(weights, alpha, predictions, y):
#     factor = np.exp(-alpha * y * predictions)
#     weights *= factor / np.sum(weights)
#
#
# def adaboost(X, y, num_classifiers):
#     num_samples, _ = X.shape
#     weights = np.ones(num_samples) / num_samples
#     classifiers = []
#
#     for _ in range(num_classifiers):
#         stump, error = train_decision_stump(X, y, weights)
#         alpha = 0.5 * np.log((1 - error) / max(error, 1e-10))
#         stump.alpha = alpha
#         classifiers.append(stump)
#
#         predictions = stump.prediction
#         update_weights(weights, alpha, predictions, y)
#
#     return classifiers
#
#
# def adaboost_predict(X, classifiers):
#     predictions = np.zeros(X.shape[0])
#     for stump in classifiers:
#         threshold = stump.threshold
#         feature_index = stump.feature_index
#         alpha = stump.alpha
#         prediction = stump.prediction
#
#         predictions += alpha * (X[:, feature_index] <= threshold) * prediction
#
#     return np.sign(predictions)
#
#
# if __name__ == '__main__':
#     # Example usage with the provided 3D matrix
#     data = [[[0, 0, 0, 0], 'neg'], [[1, 0, 0, 0], 'pos'], [[1, 0, 0, 1], 'neg'], [[0, 0, 0, 1], 'neg']]
#     X = np.array([x[0] for x in data])
#     y = np.array([1 if x[1] == 'pos' else -1 for x in data])
#
#     # Convert labels to 1 and -1, and ensure X is float
#     y = np.array([1 if label == 'pos' else -1 for _, label in data])
#     X = X.astype(float)
#
#     num_classifiers = 3
#     classifiers = adaboost(X, y, num_classifiers)
#
#     # Example prediction
#     test_data = np.array([[1, 0, 0, 0], [0, 0, 0, 1]])
#     predictions = adaboost_predict(test_data, classifiers)
#
#     print("Predictions:", predictions)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


class Node:
    def __init__(self, feature_index=None, threshold=None, data_left=None, data_right=None, gain=None, value=None,
                 counting=None, current_depth=None, node_position=None, node_prediction=None):
        # Decision Node
        if node_prediction is None:
            node_prediction = []
        self.feature_index = feature_index
        self.threshold = threshold
        self.data_left = data_left
        self.data_right = data_right
        self.gain = gain

        # Leaf Node
        self.value = value

        # For plot
        self.counting = counting
        self.current_depth = current_depth
        self.node_position = node_position
        self.node_prediction = node_prediction


class DecisionTreeMethod:
    def __init__(self, min_split=2, max_depth=5, n_features=None, final_depth=None, final_left=None, final_right=None):
        # Initial root
        self.root = None

        # Stopping conditions
        self.min_split = min_split
        self.max_depth = max_depth

        # Number of features
        self.n_features = n_features

        # For plot axis
        self.final_depth = final_depth
        self.final_left = final_left
        self.final_right = final_right

    @staticmethod
    def _dictionary_sort(dictionary):
        # Sorts the data in each node to be displayed in order and the same in each node
        list_to_sort = []
        sorted_dict = {}
        for key, value in dictionary.items():
            list_to_sort.append(int(key))
        list_to_sort.sort()
        for i in list_to_sort:
            for key, value in dictionary.items():
                if i == key:
                    sorted_dict[i] = value
        return sorted_dict

    @staticmethod
    # Function will return an array with that has counted the number of different elements when most_common == false.
    # Function will return the most common element when most_common == true.
    def _counter(s, most_common=False, plot_id=False):
        s_dict = {}
        for i in s:
            if i not in s_dict:
                s_dict[i] = 1
            else:
                s_dict[i] += 1
        s_dict = model._dictionary_sort(s_dict)
        if plot_id:
            return s_dict
        if not most_common:
            s_array = np.array(list(s_dict.values()))
            return s_array
        else:
            # The most common value in node
            s_mode = max(s_dict, key=s_dict.get)
            return s_mode

    def entropy(self, s):
        # Shannon Entropy calculated (-sum p(x)log_2(p(x)))
        probability_array = self._counter(s)
        probability_array = probability_array / len(s)

        return -np.sum(probability_array * np.log2(probability_array))

    def information_gain(self, parent, l_child, r_child):
        # KL divergence
        left_value = len(l_child) / len(parent)
        right_value = len(r_child) / len(parent)

        return self.entropy(parent) - (left_value * self.entropy(l_child) + right_value * self.entropy(r_child))

    def best_fit(self, X, y):
        m_samples, n_features = X.shape
        # Create empty dict to store possible best fit
        split_dict = {}
        best_gain = -1

        # Cycle through features to find the best information gain
        for feature_index in range(n_features):
            X_column = X[:, feature_index]
            for threshold in np.unique(X_column):
                data = np.concatenate((X, y.reshape(1, -1).T), axis=1)
                data_left = np.array([sample for sample in data if sample[feature_index] <= threshold])
                data_right = np.array([sample for sample in data if sample[feature_index] > threshold])
                if len(data_left) > 0 and len(data_right) > 0:
                    y_left = data_left[:, -1]
                    y_right = data_right[:, -1]

                    current_info_gain = self.information_gain(y, y_left, y_right)
                    if current_info_gain > best_gain:
                        best_gain = current_info_gain
                        split_dict = {
                            'feature index': feature_index,
                            'threshold': threshold,
                            'gain': best_gain,
                            'leaf left': data_left,
                            'leaf right': data_right
                        }
        return split_dict

    def tree_growth(self, X, y, depth=0, node_position=0):
        m_samples, n_features = X.shape
        # Position of node
        diff_sum = 0
        if node_position is not None:
            diff_depth = self.max_depth - depth
            diff_list = []
            for i in range(1, diff_depth + 1):
                diff_list.append(i)
            diff_sum = sum(diff_list)

        else:
            node_position = node_position

        # Determine the depth of the tree for use in plotting
        if self.final_depth is not None:
            if depth > self.final_depth:
                self.final_depth = depth
        else:
            self.final_depth = depth
        if self.final_left is not None:
            if node_position < self.final_left:
                self.final_left = node_position
        else:
            self.final_left = node_position
        if self.final_right is not None:
            if node_position > self.final_right:
                self.final_right = node_position
        else:
            self.final_right = node_position

        # Stopping criteria
        if depth >= self.max_depth or len(np.unique(y)) == 1 or m_samples < self.min_split:
            # If criteria is met then the node is a leaf node
            leaf_value = self._counter(y, most_common=True)
            node_occur = self._counter(y, plot_id=True)
            return Node(value=leaf_value, counting=node_occur, current_depth=depth,
                        node_position=node_position)

        best_feature = self.best_fit(X, y)
        if best_feature['gain'] > 0:
            # Left of tree
            left = self.tree_growth(X=best_feature['leaf left'][:, :-1],
                                    y=best_feature['leaf left'][:, -1],
                                    depth=depth + 1, node_position=node_position - diff_sum)
            # Right of tree
            right = self.tree_growth(X=best_feature['leaf right'][:, :-1],
                                     y=best_feature['leaf right'][:, -1],
                                     depth=depth + 1, node_position=node_position + diff_sum)
            node_occur = self._counter(y, plot_id=True)
            return Node(feature_index=best_feature['feature index'],
                        threshold=best_feature['threshold'],
                        data_left=left,
                        data_right=right,
                        gain=best_feature['gain'],
                        counting=node_occur,
                        current_depth=depth,
                        node_position=node_position)

    def fit(self, X, y):
        self.root = self.tree_growth(X, y)

    def _predict(self, x, tree):
        # Leaf
        if tree.value is not None:
            tree.node_prediction.append(x[-1])
            return tree.value
        feature_value = x[tree.feature_index]

        # Left
        if feature_value <= tree.threshold:
            tree.node_prediction.append(x[-1])
            return self._predict(x=x, tree=tree.data_left)

        # Right
        if feature_value > tree.threshold:
            tree.node_prediction.append(x[-1])
            return self._predict(x=x, tree=tree.data_right)

    def predict(self, X, y):
        test_data = np.hstack((X, y.reshape(y.shape[0], 1)))
        return [self._predict(x, self.root) for x in test_data]

    @staticmethod
    def data_names(names, index_value):
        # Used to extract feature and target names used for showing data in plot
        for i in names:
            name_index = np.where(names == i)[0]
            if name_index == index_value:
                return names[index_value]

    def plot_tree(self, axis, tree=None, previous_position=None, previous_depth=None):
        # Plotting tree using data saved into the node info.
        if not tree:
            tree = self.root
        # Plot lines between nodes
        # print(tree.node_prediction)
        if previous_position is not None:
            plt.plot([previous_position * 25, tree.node_position * 25],
                     [self.final_depth - previous_depth, self.final_depth - tree.current_depth],
                     color='k')
        # Get data from test predictions
        if len(tree.node_prediction) == 0:
            node_prediction_results = None
        else:
            node_prediction_results = self._counter(tree.node_prediction, plot_id=True)
        # Create leaf nodes
        if tree.value is not None:
            node_info_string = str(model.data_names(y_names, tree.value)) + "\n" + "train:" + str(tree.counting) + \
                               "\n" + "test:" + str(node_prediction_results)
            axis.text(tree.node_position*25, self.final_depth - tree.current_depth, node_info_string,
                      color='k', bbox=dict(facecolor='lightgreen', edgecolor='k', boxstyle='round'), ha='center',
                      va='center', fontsize='xx-small')
        # Or else create root nodes
        else:
            node_info_string = str(model.data_names(X_names, tree.feature_index)) + "\n" + " <= " + str(tree.threshold)\
                               + "\n" + "gain:" + str(round(tree.gain, 5)) + "\n" + "train:" + str(tree.counting) + \
                               "\n" + "test:" + str(node_prediction_results)
            ax.text(tree.node_position*25, self.final_depth - tree.current_depth, node_info_string,
                    color='k', bbox=dict(facecolor='lightyellow', edgecolor='k', boxstyle='round'),
                    ha='center', va='center', fontsize='xx-small')
            self.plot_tree(axis, tree.data_left,
                           previous_position=tree.node_position, previous_depth=tree.current_depth)
            self.plot_tree(axis, tree.data_right,
                           previous_position=tree.node_position, previous_depth=tree.current_depth)


if __name__ == "__main__":
    # Load data
    iris = load_iris()
    X_data = iris['data']
    y_target = iris['target']

    # Save feature and target names for convenience
    X_names = np.array(iris['feature_names'])
    y_names = iris['target_names']

    # Split the data into training and test data (80/20 respectively)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_target, test_size=0.2, random_state=9)

    # Train model and predict results
    model = DecisionTreeMethod()
    model.fit(X_train, y_train)
    predicted_results = model.predict(X_test, y_test)

    # Calculate accuracy
    result = accuracy_score(y_test, predicted_results)

    # Plot decision tree
    fig, ax = plt.subplots()
    ax.set(xlim=(model.final_left * 25 - 1, model.final_right * 25 + 1), ylim=(-1, model.final_depth + 1))
    model.plot_tree(ax)

    # Show accuracy on plot
    ax.text(model.final_right * 24, -1, "Accuracy = {}%".format(result * 100),
            color='k', bbox=dict(facecolor='white', edgecolor='white'), ha='center', va='center')

    plt.axis('off')
    plt.show()

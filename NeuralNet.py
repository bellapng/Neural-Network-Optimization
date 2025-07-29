# Isabella Castillo, NetID: iac240000
# Mariam Hamza, NetID: mxh230045

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
from ucimlrepo import fetch_ucirepo

# To silence all the convergence warnings when testing hyperparams
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning, module='sklearn')

class NeuralNet:

    # Fetches the 'Breast Cancer Wisconsin (Diagnostic)' dataset
    def __init__(self):
        print("\nFetching dataset...")
        breast_cancer = fetch_ucirepo(id=17)
        X = breast_cancer.data.features
        y = breast_cancer.data.targets
        self.original_data = pd.concat([X, y], axis=1)


    # Preprocess with handling null/empty values, ensuring data integrity, and standardization
    def preprocess(self):
        print("\nPreprocessing the dataset...")
        df = self.original_data.copy()

        # Dropping any rogue missing values (dataset is defined on the repo as having none, so we will just be sure of it)
        df.dropna(inplace=True)

        # Encoding the label via map (no use in creating separate dummies df for one-hot encoding and then concat-ing)
        label = df.columns[-1]
        df[label] = df[label].map({'M': 1, 'B': 0})

        # Split into separate dfs to standardize continuous values
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Standardize values and assigning column names to the scaled feature set
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        self.processed_data = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
        return self.processed_data


    # Model training and evaluation phase
    def train_evaluate(self):

        print("\nTraining the neural network...")
        ncols = len(self.processed_data.columns)
        X = self.processed_data.iloc[:, :ncols-1]
        y = self.processed_data.iloc[:, ncols-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        # Below are the hyperparameters that we will use for model evaluation, assuming any fixed number of neurons for each hidden layer
        activations = ['logistic', 'tanh', 'relu']
        learning_rate = [0.001, 0.01] # added 0.001, took away 0.1
        max_iterations = [100, 200]
        hidden_layer_options = [

            # 1 hl, 2hl, then 3hl
            (30,), (15, 15), (10, 10, 10)
        ]

        # Creating the neural network and keeping track of results for report and plotting (3 plots for better readability)
        results = []
        losses_1hl = []
        losses_2hl = []
        losses_3hl = []

        # Iterating over all hyperparameter combos
        for acti_func in activations:
            for lr in learning_rate:
                for max_iter in max_iterations:
                    for num_hl in hidden_layer_options:

                        # For plot naming later
                        label = "acti_func=" + str(acti_func) + ", lr=" + str(lr) + ", max_iter=" + str(max_iter) + ", num_hl=" + str(num_hl)

                        # Traininhg the MLP Classifier
                        model = MLPClassifier(activation=acti_func, learning_rate_init=lr, max_iter=max_iter, hidden_layer_sizes=num_hl, random_state=42)
                        model.fit(X_train, y_train)

                        # Computing the metrics for loss
                        y_test_proba = model.predict_proba(X_test)
                        train_loss = model.loss_curve_[-1]
                        test_loss = log_loss(y_test, y_test_proba)

                        # Recording results for later to compare
                        results.append({
                            'activation': acti_func,
                            'learning_rate': lr,
                            'max_iter': max_iter,
                            'layers': num_hl,
                            'train_loss': train_loss,
                            'test_loss': test_loss
                        })

                        hidden_layer_count = len(num_hl)
                        if hidden_layer_count == 1:
                            losses_1hl.append((model.loss_curve_, label, max_iter))
                        if hidden_layer_count == 2:
                            losses_2hl.append((model.loss_curve_, label, max_iter))
                        if hidden_layer_count == 3:
                            losses_3hl.append((model.loss_curve_, label, max_iter))

        # Plotting the model history for each model in a single plot model history is a plot of accuracy vs number of epochs you may want to create a large sized plot to show multiple lines in a same figure.
        print("\nGenerating model history plots...")

        if losses_1hl:
            plt.figure(figsize=(18,12))
            num_colors_1hl = len(losses_1hl)
            cmap = plt.colormaps['turbo']
            colors = [cmap(i) for i in np.linspace(0, 1, num_colors_1hl)]

            for i, (history, current_label_string, current_max_iter) in enumerate(losses_1hl):
                curr_marker = None
                curr_linestyle = '-'
                if current_max_iter == 100:
                    curr_marker = 'o'
                    curr_linestyle = '-'
                elif current_max_iter == 200:
                    curr_marker = '^'
                    curr_linestyle = '--'
                plt.plot(history, label=current_label_string, color=colors[i], marker=curr_marker, linestyle=curr_linestyle, markevery=20)

            plt.title('Model Training Loss vs # of Epochs (1 Hidden Layer)')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
            plt.tight_layout()
            plt.show()

        if losses_2hl:
            plt.figure(figsize=(18,12))
            num_colors_2hl = len(losses_2hl)
            cmap = plt.colormaps['turbo']
            colors = [cmap(i) for i in np.linspace(0, 1, num_colors_2hl)]

            for i, (history, current_label_string, current_max_iter) in enumerate(losses_2hl):
                curr_marker = None
                curr_linestyle = '-'
                if current_max_iter == 100:
                    curr_marker = 'o'
                    curr_linestyle = '-'
                elif current_max_iter == 200:
                    curr_marker = '^'
                    curr_linestyle = '--'
                plt.plot(history, label=current_label_string, color=colors[i], marker=curr_marker, linestyle=curr_linestyle, markevery=20)

            plt.title('Model Training Loss vs # of Epochs (2 Hidden Layers)')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
            plt.tight_layout()
            plt.show()

        if losses_3hl:
            plt.figure(figsize=(18,12))
            num_colors_3hl = len(losses_3hl)
            cmap = plt.colormaps['turbo']
            colors = [cmap(i) for i in np.linspace(0, 1, num_colors_3hl)]

            for i, (history, current_label_string, current_max_iter) in enumerate(losses_3hl):
                curr_marker = None
                curr_linestyle = '-'
                if current_max_iter == 100:
                    curr_marker = 'o'
                    curr_linestyle = '-'
                elif current_max_iter == 200:
                    curr_marker = '^'
                    curr_linestyle = '--'
                plt.plot(history, label=current_label_string, color=colors[i], marker=curr_marker, linestyle=curr_linestyle, markevery=20)

            plt.title('Model Training Loss vs # of Epochs (3 Hidden Layers)')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
            plt.tight_layout()
            plt.show()

        df_results = pd.DataFrame(results)
        df_sorted = df_results.sort_values('test_loss', ascending=True)
        return df_sorted


if __name__ == "__main__":

    neural_network = NeuralNet()
    neural_network.preprocess()
    df_results = neural_network.train_evaluate()

    print("\nModel Evaluation Results")
    print(df_results)

    # For exporting to table in report later, will comment out
    #df_results.to_csv('neural_network_results.csv', index=False)


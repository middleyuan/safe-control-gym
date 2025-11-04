import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


class LinearModel:
    '''
    multiple input, single output linear regression model
    '''

    def __init__(self,
                 input_data,
                 target_data,
                 feature: cs.Function):
        self.num_data = input_data.shape[0]
        self.input_data = input_data
        self.target_data = target_data

        self.feature = feature
        self.feature_data = np.array([self.feature(self.input_data[i])
                                      for i in range(self.num_data)]).squeeze()
        print(f'feature_data shape: {self.feature_data.shape}')
        self.model = LinearRegression()

    def fit(self):
        # fit the model
        self.model.fit(self.feature_data, self.target_data)
        # report the score
        score = self.model.score(self.feature_data, self.target_data)
        print(f'fitting score: {score}')
        print(f'weight: {self.model.coef_}')
        return self.weight

    def weight(self):
        return self.model.coef_

    def predict(self, input_data):
        feature_data = np.array([self.feature(input_data[i])
                                 for i in range(input_data.shape[0])]).squeeze()
        return self.model.predict(feature_data)


if __name__ == '__main__':
    num_data = 10
    input_data = np.linspace(0, np.pi, num_data).reshape(-1, 1)
    target_data = 1.5 * np.sin(input_data).reshape(-1, 1)

    # Define feature
    x = cs.MX.sym('x', 1, 1)
    feature = cs.Function('feature', [x], [cs.vertcat(1, x, cs.sin(x))])

    model = LinearModel(input_data, target_data, feature)
    model.fit()

    # Plot the results
    fig, ax = plt.subplots()
    ax.scatter(input_data, target_data, label='data', color='r')
    ax.plot(input_data, model.predict(input_data), label='prediction')
    ax.legend()
    plt.show()

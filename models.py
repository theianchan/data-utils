import numpy as np

def root_mean_squared_error(y_pred, y):
    """Calculates root mean squared error"""

    return np.sqrt(np.mean((y_pred - y) ** 2))


def lm_evaluate(lm, x=None, y=None):
    """Takes a linear model and prints the intercept,
    coefficients, R-squared value, and RMSE

    Returns R-squared value and RMSE in a tuple"""

    print("Intercept: {}\nCoefficients: {}".format(
        lm.intercept_, lm.coef_
    ))
    if (x is not None) and (y is not None):
        r_squared = lm.score(x, y)
        rmse = root_mean_squared_error(lm.predict(x), y)
        print("R-Squared Value: {}\nRMSE: {}".format(
            r_squared, rmse
        ))
        return (r_squared, rmse)

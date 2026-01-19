from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

def load_data(test_size=0.2):
    X, y = load_diabetes(return_X_y=True)
    y = (y > y.mean()).astype(int)

    return train_test_split(
        X, y, test_size=test_size, random_state=42
    )

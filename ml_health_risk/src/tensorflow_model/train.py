from .model import build_model

def train_tf(X_train, y_train):
    model = build_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=1)
    return model

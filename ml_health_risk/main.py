from src.data_loader import load_data
from src.preprocessing import preprocess
from src.tensorflow_model.train import train_tf

X_train, X_test, y_train, y_test = load_data()
X_train, X_test = preprocess(X_train, X_test)

model = train_tf(X_train, y_train)
loss, acc = model.evaluate(X_test, y_test)

print(f"Test Accuracy: {acc:.2f}")

from keras.models import model_from_json
from keras.utils import plot_model

# ファイルからモデルを読み込み
print("Loading model...")
model = None
with open("./drive/My Drive/model.json") as f:
  model = model_from_json(f.read())

# ファイルから重みを読み込み
print("Loading weights...")
model.load_weights("./drive/My Drive/weights.hdf5")

print("Predict test data...")
y_pred = model.predict(X_test, verbose=1)
print("complete.")

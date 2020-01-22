# CNNを構築
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2)) # クラスは2個
model.add(Activation('softmax'))

# コンパイル
print("Model compiling...")
model.compile(loss='binary_crossentropy',
              optimizer='SGD',
              metrics=['accuracy'])

# 実行。出力はありで設定(verbose=1)。
print("fitting...")
history = model.fit(X_train, Y_train, batch_size=30, epochs=10,
                   validation_split=0.2, shuffle=True, verbose = 1)

# モデルを保存
print("Model saving... at ./drive/My Drive/model.json")
from keras.utils import plot_model
model_json = model.to_json()
with open("./drive/My Drive/model.json", mode='w') as f:
  f.write(model_json)

# 学習済みの重みを保存
print("Weight saving... at ./drive/My Drive/weights.hdf5")
model.save_weights("./drive/My Drive/weights.hdf5")

# 精度をプロットして表示
print("plotting...")
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['acc', 'val_acc'], loc='lower right')
plt.show()

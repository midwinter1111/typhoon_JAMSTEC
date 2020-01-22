# 訓練データを読み込む

X_train = []
Y_train = []

notify_steps = 1000 # 1000ファイルごとに進捗を出力

# 正例
print("Reading True images...")
files = glob.glob("./train/True/*.tif")
number = 0
for picture in files:
  tmp_img = Image.open(picture)
  img = img_to_array(tmp_img)
  X_train.append(img)
  Y_train.append(1)
  number+=1
  if number % notify_steps == 0:
    print("Reading "+str(number)+" / "+str(len(files))+"True images...")

# 負例
print("Reading False images...")
files = glob.glob("./train/False/*.tif")
number = 0
for picture in files:
  tmp_img = Image.open(picture)
  img = img_to_array(tmp_img)
  X_train.append(img)
  Y_train.append(0)
  number+=1
  if number % notify_steps == 0:
    print("Reading "+str(number)+" / "+str(len(files))+"False images...")

# arrayに変換
X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)

# クラスの形式を変換: One-Hot表現
Y_train = np_utils.to_categorical(Y_train, 2) 

print("Complete.")  

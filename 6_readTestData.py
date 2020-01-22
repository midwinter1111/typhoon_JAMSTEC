# テストデータzipファイルの展開
# テストデータの読み込み

X_test = []

notify_steps = 100 # 100ファイルごとに進捗を出力

print("Reading test images...")
files = glob.glob("./test/*.tif")

# ファイルリストをソート
files_test = sorted(files)

number = 0
for picture in files_test:
  tmp_img = Image.open(picture)
  img = img_to_array(tmp_img)
  X_test.append(img)
  number += 1
  if number % notify_steps == 0:
    print("Reading "+str(number)+" / "+str(len(files_test))+"test images...")

# arrayに変換
X_test = np.asarray(X_test)

print("complete.")

# TSVファイルの作成
print("create TSV file...")
path = './drive/My Drive/predict.tsv'
with open(path, mode='w') as f:
  for i in range(len(files_test)):
    tmp_f = files_test[i].replace("./test/", "")
    tmp_p = "0"
    if y_pred[i][0] > 0.5:
      tmp_p = "0"
    else:
      tmp_p = "1"
    out_str = tmp_f + '\t' + tmp_p + '\n'
    f.write(out_str)

print("complete.")

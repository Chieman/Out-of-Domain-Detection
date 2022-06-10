from vncorenlp import VnCoreNLP
import pandas as pd
import csv

annotator = VnCoreNLP("/home/an/Documents/out-of-domain/Coding3/VnCoreNLP/VnCoreNLP-1.1.1.jar",annotators="wseg")
data = pd.read_csv("/home/an/Documents/out-of-domain/Coding3/Untitled spreadsheet - Sheet1.csv")
text_data = data['text']
label = data['label']
processed_data = open('/home/an/Documents/out-of-domain/Coding3/WSUntitled spreadsheet - Sheet1.csv', 'w')
writer = csv.writer(processed_data, delimiter=",")
writer.writerow((str("text"), str("labels")))
for text_id in range(len(text_data)):
  word_segmented_text = annotator.tokenize(text_data[text_id])
  new_text = word_segmented_text[0][0]
  for i in range(len(word_segmented_text)):
    if i == 0:
      for j in range(1, len(word_segmented_text[i])):
        new_text += " "
        new_text += word_segmented_text[i][j]
    else:
      for j in range(len(word_segmented_text[i])):
        new_text += " "
        new_text += word_segmented_text[i][j]

  writer.writerow((new_text, label[text_id]))
processed_data.close()

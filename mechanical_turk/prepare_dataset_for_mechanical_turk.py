import pandas as pd
import random
from encode_emoji import replace_emoji_characters
random.seed(42)

df = pd.read_json('../i_am_a_strange_dataset.jsonl', lines=True)

turk_formatted_1 = []
turk_formatted_2 = []
ids_1 = []
ids_2 = []
gold_labels_1 = []
gold_labels_2 = []
for index, row in df.iterrows():
    false_sentence = replace_emoji_characters(row["beginning"] + row["false_continuation"])
    true_sentence = replace_emoji_characters(row["beginning"] + row["true_continuation"])
    true_and_false_sentences = [true_sentence, false_sentence]
    random.shuffle(true_and_false_sentences)
    if true_and_false_sentences[0] == true_sentence:
        gold_labels_1.append(True)
        gold_labels_2.append(False)
    else:
        gold_labels_1.append(False)
        gold_labels_2.append(True)
    ids_1.append(row["id"])
    ids_2.append(row["id"])
    turk_formatted_1.append(f"<pre style='overflow: scroll; max-height: 300px;'>{true_and_false_sentences[0]}</pre>" if "\n" in true_and_false_sentences[0] else true_and_false_sentences[0])
    turk_formatted_2.append(f"<pre style='overflow: scroll; max-height: 300px;'>{true_and_false_sentences[1]}</pre>" if "\n" in true_and_false_sentences[1] else true_and_false_sentences[1])

df_turk_formatted_1 = pd.DataFrame({"text": turk_formatted_1, "id": ids_1, "gold_label": gold_labels_1})
df_turk_formatted_2 = pd.DataFrame({"text": turk_formatted_2, "id": ids_2, "gold_label": gold_labels_2})

df_turk_formatted_1.to_csv('turk_formatted_1_i_am_a_strange_dataset.csv', index=False)
df_turk_formatted_2.to_csv('turk_formatted_2_i_am_a_strange_dataset.csv', index=False)



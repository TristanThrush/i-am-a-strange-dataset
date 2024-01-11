from cryptography.fernet import Fernet
import os

def decrypt_file(input_filename, output_filename, key):
    fernet = Fernet(key)

    with open(input_filename, 'rb') as encrypted_file:
        encrypted_data = encrypted_file.read()

    decrypted_data = fernet.decrypt(encrypted_data)

    with open(output_filename, 'wb') as decrypted_file:
        decrypted_file.write(decrypted_data)

answer = input("""
Welcome!

If you want to decrypt the dataset files in this repo, you must be a human.
You must also promise not to use the data for model training (only for evaluation).
      
If you agree, then successfully complete the following self-referential statement.
Answer either "Yes" or "No" (pay attention to the capitalization):

"if someone asks whether this sentence has a capital letter, the correct answer is"
""")

if answer == "Yes":
    encryption_key = answer + "wvdOH2jSfaFoSYoyM6_a0U3BHw43HjSC4Uhzav6k="
elif answer == "No":
    encryption_key = answer + "swvdOH2jSfaFoSYoyM6_a0U3BHw43HjSC4Uhzav6k="
else:
    raise ValueError("Please enter either Yes or No.")

encryption_key = encryption_key.encode('utf-8')

decrypt_file("encrypted_i_am_a_strange_dataset.jsonl", "i_am_a_strange_dataset.jsonl", encryption_key)
decrypt_file("encrypted_i_am_an_impossible_dataset.jsonl", "i_am_an_impossible_dataset.jsonl", encryption_key)

decrypt_file("mechanical_turk/encrypted_turk_formatted_1_i_am_a_strange_dataset.csv", "mechanical_turk/turk_formatted_1_i_am_a_strange_dataset.csv", encryption_key)
decrypt_file("mechanical_turk/encrypted_turk_formatted_2_i_am_a_strange_dataset.csv", "mechanical_turk/turk_formatted_2_i_am_a_strange_dataset.csv", encryption_key)

decrypt_file("mechanical_turk/results/encrypted_Batch_5171265_batch_results.csv", "mechanical_turk/results/Batch_5171265_batch_results.csv", encryption_key)
decrypt_file("mechanical_turk/results/encrypted_Batch_5171266_batch_results.csv", "mechanical_turk/results/Batch_5171266_batch_results.csv", encryption_key)

if not os.path.exists("evaluation_results"):
    os.mkdir("evaluation_results")
for root, dirs, files in os.walk("encrypted_evaluation_results"):
    for file in files:
        decrypt_file(os.path.join(root, file), os.path.join("evaluation_results", file), encryption_key)

if not os.path.exists("impossible_evaluation_results"):
    os.mkdir("impossible_evaluation_results")
for root, dirs, files in os.walk("encrypted_impossible_evaluation_results"):
    for file in files:
        decrypt_file(os.path.join(root, file), os.path.join("impossible_evaluation_results", file), encryption_key)
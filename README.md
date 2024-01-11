# I am a Strange Dataset

This is the repository for "I am a Strange Dataset: Metalinguistic Tests for Language Models".

```
@article{thrush2024strange,
    author    = {Tristan Thrush and Jared Moore and Miguel Monares and Christopher Potts and Douwe Kiela},
    title     = {I am a Strange Dataset: Metalinguistic Tests for Language Models},
    journal   = {arXiv},
    year      = {2024},
}
```

Here, you can find full model outputs used in the statistics from the paper. You can also use the code here to evaluate a variety of models on the Hugging Face Hub or private models via APIs.

Additionally, we provide the code that we used to compute tables and graphs from the paper.

Of course, you can also find the dataset itself here.

## Setup

First things first:

```
git clone https://github.com/TristanThrush/i-am-a-strange-dataset.git
cd i-am-a-strange-dataset
pip install -r requirements.txt
```

Then decrypt the dataset and model output files by correctly completing a self-referential statement (the answer is part of the encryption key).

```
python decrypt.py
```

## Running Evaluations

Evaluation results for all models in the paper are already saved in the `evaluation_results` directory. 
Although, you can run more evaluations yourself, using the `evaluate.py` script.
See `evaluate_open_source_models_stanford_cluster.sh` for an example of how to use it to evaluate models from the Hugging Face Hub.
You can also use it to evaluate supported API models. For example, to evaluate GPT-4, you would run:

```
python evaluate.py --model gpt-4 --api_model --api_has_logprobs
```

To evaluate API models, you need to have the relevant API keys in an `.env` file. See `example.env` for an example.

## Computing Statistics

After running the evaluation, result files are saved in the `evaluation_results` dir.
To display the overall scores, confidence intervals, and scaling plots for the model evaluation files, you can run:

```
python compute_statistics.py
```

## Running the Human Experiment on Amazon Mechanical Turk

In order to ensure that "I am a strange dataset" is understandable by humans,
we run the human equivalent of the validation task on Amazon Mechanical Turk.

The following code randomly splits the pairs of (true, false) sentences from the dataset into two different files. It also encodes line breaks and emojis into a format that Mechanical Turk's CSV loader can understand.

```
cd mechanical_turk
python prepare_dataset_for_mechanical_turk.py
```

After uploading the CSVs to Mechanical Turk, you also need to follow [these instructions](https://stackoverflow.com/questions/76882907/how-to-prepare-a-csv-hit-file-containing-emoji-character-for-amazon-mechanical-t) for your UI to decode the emojis correctly.

Note that we get 10 Turker judgements for each sentence, and we select Turkers that satisfy the following criteria:
* Master
* US only
* 1000 or more previous HITs approved
* 95% or higher HIT approval rate
* Custom qualification test (as seen in the `mechanical_turk` directory)

To launch the qualification test via `launch_qualification_test.py`, you need to have the relevant API keys in an `.env` file. See `example.env` for an example.

The instructions are simply:

```
Say whether the following self-referential statement is true or false.
```

The detailed instructions are the same as the few shot instructions given to the models (with an additional paragraph for the humans about not using AI):

```
Overview:

Your task is to say whether a self-referential statement is true or false. In cases where you think that the answer is ambiguous, you should use your best judgement and still say either 'True' or 'False'.

Do not use any AI assistants such as ChatGPT to help you; AI assistants perform very poorly at this task and so will get many of the answers wrong. Although, you can otherwise feel free to search online for any information that would help you answer confidently. For example, a few statements may contain a language besides English. So, you can feel free to use Google Translate. You can also search for the definitions of words that you are unfamiliar with.

Make sure that you understand the given examples before continuing.


Example:

I am the text you are reading.
My last line has how many words?
Two words.

Answer: True


Example:

I am the text you are reading.
My last line has how many words?
Seven words.

Answer: False


Example:

this sentence
l             o
o             o
k             s
a bit like the
l
e
t
t
e
r
"p"

Answer: True


Example:

this sentence
l             o
o             o
k             s
a bit like the
l
e
t
t
e
r
"o"

Answer: False


Example:

here in reversed is order word the.

Answer: True


Example:

here in fine is order word the.

Answer: False
```

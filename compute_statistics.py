import os
import json
import pandas as pd
import scipy.stats
import math


def name_to_pretty_name_parameters_chat_order(name):
    name = name.replace("evaluation_results/", "")
    if name == "Llama-2-7b-hf":
        return "Llama 2", "7B", "N", 0
    elif name == "Llama-2-7b-chat-hf":
        return "Llama 2", "7B", "Y", 1
    elif name == "Mistral-7B-v0.1":
        return "Mistral 0.1", "7B", "N", 2
    elif name == "Starling-LM-7B-alpha":
        return "Starling $\\alpha$", "7B", "Y", 3
    elif name == "Mistral-7B-Instruct-v0.2":
        return "Mistral 0.2", "7B", "Y", 4
    elif name == "Llama-2-13b-hf":
        return "Llama 2", "13B", "N", 5
    elif name == "Llama-2-13b-chat-hf":
        return "Llama 2", "13B", "Y", 6
    elif name == "Mixtral-8x7B-v0.1":
        return "Mixtral 0.1", "8x7B", "N", 7
    elif name == "Mixtral-8x7B-Instruct-v0.1":
        return "Mixtral 0.1", "8x7B", "Y", 8
    elif name == "Llama-2-70b-hf":
        return "Llama 2", "70B", "N", 9
    elif name == "Llama-2-70b-chat-hf":
        return "Llama 2", "70B", "Y", 10
    elif name == "claude-2":
        return "Claude 2", "-", "Y", 11
    elif name == "gpt-3.5-turbo":
        return "GPT 3.5 T", "-", "Y", 12
    elif name == "gpt-4":
        return "GPT 4", "-", "Y", 13
    else:
        return name, "-", "-", -1


def display_number(number):
    return "{:.2f}".format(round(100*number, 2))

def display_entry(number, ci_number, random, only_bold_above_random=True): 
    if only_bold_above_random:
        bold = number > random + ci_number
    else:
        bold = number > random + ci_number or number < random - ci_number

    if bold:
        return "\\textbf{" + display_number(number) + "} $\\pm$ \\textbf{" + display_number(ci_number) + "}"
    return display_number(number) + " $\\pm$ " + display_number(ci_number)


def gen_score(true_col_name, false_col_name, scores, row):
    if row[true_col_name] < row[false_col_name]:
        scores.append(1)
    else:
        scores.append(0)

def cot_val_score(true_col_name, false_col_name, scores, row):
    
    condition_1 = "true" in row[true_col_name].lower() and "false" not in row[true_col_name].lower()

    condition_2 = "false" in row[false_col_name].lower() and "true" not in row[false_col_name].lower()

    if condition_1 and condition_2:
        scores.append(1)
    elif condition_1 and not condition_2:
        scores.append(0.5)
    elif not condition_1 and condition_2:
        scores.append(0.5)
    else:
        scores.append(0)

def rel_val_score(true_statement_true_answer_col_name, true_statement_false_answer_col_name, false_statement_true_answer_col_name, false_statement_false_answer_col_name, scores, row):
    if row[true_statement_true_answer_col_name]/row[true_statement_false_answer_col_name] < row[false_statement_true_answer_col_name]/row[false_statement_false_answer_col_name]:
        scores.append(1)
    else:
        scores.append(0)

def val_score(true_statement_true_answer_col_name, true_statement_false_answer_col_name, false_statement_true_answer_col_name, false_statement_false_answer_col_name, scores, row):
    condition_1 = row[true_statement_true_answer_col_name] < row[true_statement_false_answer_col_name]

    condition_2 = row[false_statement_false_answer_col_name] < row[false_statement_true_answer_col_name]

    if condition_1 and condition_2:
        scores.append(1)
    elif condition_1 and not condition_2:
        scores.append(0.5)
    elif not condition_1 and condition_2:
        scores.append(0.5)
    else:
        scores.append(0)

def print_results(filter_set=None, non_self_referential_comparison=True):

    def add_dash_for_missing_entries(data):
        max_len = 0
        for value in data.values():
            if len(value) > max_len:
                max_len = len(value)
        for value in data.values():
            while len(value) < max_len:
                value.append("-")

    # List all files and directories in the given path
    results_filenames = sorted([os.path.join("evaluation_results", file) for file in os.listdir("evaluation_results")])

    data = {"Model": [], "filename": [], "Params": [], "Chat": [], "Gen$^L$": [], "Val ZS$^L$": [],  "Val FS$^L$": [], "Val ZS$^L$ (R)": [], "Val FS$^L$ (R)": [], "Val CoT$^T$": []}
    data_non_self_referential_comparison = {"Model": [], "Params": [], "Chat": [], "filename": [], "$\\Delta$ Gen$^L$": [], "$\\Delta$ Val ZS$^L$": [],  "$\\Delta$ Val FS$^L$": [], "$\\Delta$ Val ZS$^L$ (R)": [], "$\\Delta$ Val FS$^L$ (R)": [], "$\\Delta$ Val CoT$^T$": []}
    for filename in results_filenames:
        model_name = filename.replace("_generation_losses.jsonl", "").replace("_validation_outputs.jsonl", "")
        if len(data["filename"]) == 0 or data["filename"][-1] != model_name:
            add_dash_for_missing_entries(data)
            data["filename"].append(model_name)

        if len(data_non_self_referential_comparison["filename"]) == 0 or data_non_self_referential_comparison["filename"][-1] != model_name:
            add_dash_for_missing_entries(data_non_self_referential_comparison)
            data_non_self_referential_comparison["filename"].append(model_name)

        if filename.endswith("_generation_losses.jsonl"):
            df = pd.read_json(filename, lines=True)
            if filter_set is not None:
                df = df[df['id'].isin(filter_set)]
            scores = []
            non_self_referential_scores = []
            non_self_referential_comparison_scores = []
            for _, row in df.iterrows():
                gen_score("true", "false", scores, row)
                if "non_self_referential_true" in row and not pd.isna(row["non_self_referential_true"]):
                    gen_score("non_self_referential_true", "non_self_referential_false", non_self_referential_scores, row)
                    gen_score("true", "false", non_self_referential_comparison_scores, row)
            bootstrap_result = scipy.stats.bootstrap(method="basic", data=(scores,), statistic=scipy.mean)
            data["Gen$^L$"].append(display_entry(sum(scores)/len(scores), (bootstrap_result.confidence_interval.high - bootstrap_result.confidence_interval.low)/2, 0.5))
            if len(non_self_referential_scores) > 1:
                non_self_referential_bootstrap_result = scipy.stats.bootstrap(method="basic", data=(non_self_referential_scores,), statistic=scipy.mean)
                data_non_self_referential_comparison["$\\Delta$ Gen$^L$"].append(display_entry(sum(non_self_referential_scores)/len(non_self_referential_scores) - sum(non_self_referential_comparison_scores)/len(non_self_referential_comparison_scores), (non_self_referential_bootstrap_result.confidence_interval.high - non_self_referential_bootstrap_result.confidence_interval.low)/2, 0, only_bold_above_random=False))
            else:
                data_non_self_referential_comparison["$\\Delta$ Gen$^L$"].append("-")
        elif filename.endswith("_validation_outputs.jsonl"):
            df = pd.read_json(filename, lines=True)
            if filter_set is not None:
                df = df[df['id'].isin(filter_set)]
            rel_scores_few_shot = []
            rel_scores_zero_shot = []
            scores_few_shot = []
            scores_zero_shot = []
            scores_cot = []

            non_self_referential_rel_scores_few_shot = []
            non_self_referential_comparison_rel_scores_few_shot = []
            non_self_referential_rel_scores_zero_shot = []
            non_self_referential_comparison_rel_scores_zero_shot = []
            non_self_referential_scores_few_shot = []
            non_self_referential_comparison_scores_few_shot = []
            non_self_referential_scores_zero_shot = []
            non_self_referential_comparison_scores_zero_shot = []
            non_self_referential_scores_cot = []
            non_self_referential_comparison_scores_cot = []
            for _, row in df.iterrows():
                if "loss_true_statement_true_answer_few_shot" in row:
                    val_score("loss_true_statement_true_answer_few_shot", "loss_true_statement_false_answer_few_shot", "loss_false_statement_true_answer_few_shot", "loss_false_statement_false_answer_few_shot", scores_few_shot, row)
                    rel_val_score("loss_true_statement_true_answer_few_shot", "loss_true_statement_false_answer_few_shot", "loss_false_statement_true_answer_few_shot", "loss_false_statement_false_answer_few_shot", rel_scores_few_shot, row)
                
                if "non_self_referential_loss_true_statement_true_answer_few_shot" in row and not pd.isna(row["non_self_referential_loss_true_statement_true_answer_few_shot"]):
                    val_score("non_self_referential_loss_true_statement_true_answer_few_shot", "non_self_referential_loss_true_statement_false_answer_few_shot", "non_self_referential_loss_false_statement_true_answer_few_shot", "non_self_referential_loss_false_statement_false_answer_few_shot", non_self_referential_scores_few_shot, row)
                    rel_val_score("non_self_referential_loss_true_statement_true_answer_few_shot", "non_self_referential_loss_true_statement_false_answer_few_shot", "non_self_referential_loss_false_statement_true_answer_few_shot", "non_self_referential_loss_false_statement_false_answer_few_shot", non_self_referential_rel_scores_few_shot, row)
                    val_score("loss_true_statement_true_answer_few_shot", "loss_true_statement_false_answer_few_shot", "loss_false_statement_true_answer_few_shot", "loss_false_statement_false_answer_few_shot", non_self_referential_comparison_scores_few_shot, row)
                    rel_val_score("loss_true_statement_true_answer_few_shot", "loss_true_statement_false_answer_few_shot", "loss_false_statement_true_answer_few_shot", "loss_false_statement_false_answer_few_shot", non_self_referential_comparison_rel_scores_few_shot, row)

                if "loss_true_statement_true_answer_zero_shot" in row:
                    val_score("loss_true_statement_true_answer_zero_shot", "loss_true_statement_false_answer_zero_shot", "loss_false_statement_true_answer_zero_shot", "loss_false_statement_false_answer_zero_shot", scores_zero_shot, row)
                    rel_val_score("loss_true_statement_true_answer_zero_shot", "loss_true_statement_false_answer_zero_shot", "loss_false_statement_true_answer_zero_shot", "loss_false_statement_false_answer_zero_shot", rel_scores_zero_shot, row)
                
                if "non_self_referential_loss_true_statement_true_answer_zero_shot" in row and not pd.isna(row["non_self_referential_loss_true_statement_true_answer_zero_shot"]):
                    val_score("non_self_referential_loss_true_statement_true_answer_zero_shot", "non_self_referential_loss_true_statement_false_answer_zero_shot", "non_self_referential_loss_false_statement_true_answer_zero_shot", "non_self_referential_loss_false_statement_false_answer_zero_shot", non_self_referential_scores_zero_shot, row)
                    rel_val_score("non_self_referential_loss_true_statement_true_answer_zero_shot", "non_self_referential_loss_true_statement_false_answer_zero_shot", "non_self_referential_loss_false_statement_true_answer_zero_shot", "non_self_referential_loss_false_statement_false_answer_zero_shot", non_self_referential_rel_scores_zero_shot, row)
                    val_score("loss_true_statement_true_answer_zero_shot", "loss_true_statement_false_answer_zero_shot", "loss_false_statement_true_answer_zero_shot", "loss_false_statement_false_answer_zero_shot", non_self_referential_comparison_scores_zero_shot, row)
                    rel_val_score("loss_true_statement_true_answer_zero_shot", "loss_true_statement_false_answer_zero_shot", "loss_false_statement_true_answer_zero_shot", "loss_false_statement_false_answer_zero_shot", non_self_referential_comparison_rel_scores_zero_shot, row)

                if "generated_text_true_statement_cot" in row:
                    cot_val_score("generated_text_true_statement_cot", "generated_text_false_statement_cot", scores_cot, row)

                if "non_self_referential_generated_text_true_statement_cot" in row and isinstance(row["non_self_referential_generated_text_true_statement_cot"], str):
                    cot_val_score("non_self_referential_generated_text_true_statement_cot", "non_self_referential_generated_text_false_statement_cot", non_self_referential_scores_cot, row)
                    cot_val_score("generated_text_true_statement_cot", "generated_text_false_statement_cot", non_self_referential_comparison_scores_cot, row)

            if len(scores_few_shot) > 1:
                bootstrap_result_few_shot = scipy.stats.bootstrap(method="basic", data=(scores_few_shot,), statistic=scipy.mean)
                data["Val FS$^L$"].append(display_entry(sum(scores_few_shot)/len(scores_few_shot), (bootstrap_result_few_shot.confidence_interval.high - bootstrap_result_few_shot.confidence_interval.low)/2, 0.5))
            else:
                data["Val FS$^L$"].append("-")
            
            if len(non_self_referential_scores_few_shot) > 1:
                result = [item1 - item2 for item1, item2 in zip(non_self_referential_scores_few_shot, non_self_referential_comparison_scores_few_shot)]
                bootstrap_result = scipy.stats.bootstrap(method="basic", data=(result,), statistic=scipy.mean)
                data_non_self_referential_comparison["$\\Delta$ Val FS$^L$"].append(display_entry(sum(result)/len(result), (bootstrap_result.confidence_interval.high - bootstrap_result.confidence_interval.low)/2, 0, only_bold_above_random=False))
            else:
                data_non_self_referential_comparison["$\\Delta$ Val FS$^L$"].append("-")

            if len(rel_scores_few_shot) > 1:
                bootstrap_result_rel_few_shot = scipy.stats.bootstrap(method="basic", data=(rel_scores_few_shot,), statistic=scipy.mean)
                data["Val FS$^L$ (R)"].append(display_entry(sum(rel_scores_few_shot)/len(rel_scores_few_shot), (bootstrap_result_rel_few_shot.confidence_interval.high - bootstrap_result_rel_few_shot.confidence_interval.low)/2, 0.5))
            else:
                data["Val FS$^L$ (R)"].append("-")

            if len(non_self_referential_rel_scores_few_shot) > 1:
                result = [item1 - item2 for item1, item2 in zip(non_self_referential_rel_scores_few_shot, non_self_referential_comparison_rel_scores_few_shot)]
                bootstrap_result = scipy.stats.bootstrap(method="basic", data=(result,), statistic=scipy.mean)
                data_non_self_referential_comparison["$\\Delta$ Val FS$^L$ (R)"].append(display_entry(sum(result)/len(result), (bootstrap_result.confidence_interval.high - bootstrap_result.confidence_interval.low)/2, 0, only_bold_above_random=False))
            else:
                data_non_self_referential_comparison["$\\Delta$ Val FS$^L$ (R)"].append("-")

            if len(scores_zero_shot) > 1:
                bootstrap_result_zero_shot = scipy.stats.bootstrap(method="basic", data=(scores_zero_shot,), statistic=scipy.mean)
                data["Val ZS$^L$"].append(display_entry(sum(scores_zero_shot)/len(scores_zero_shot), (bootstrap_result_zero_shot.confidence_interval.high - bootstrap_result_zero_shot.confidence_interval.low)/2, 0.5))
            else:
                data["Val ZS$^L$"].append("-")

            if len(non_self_referential_scores_zero_shot) > 1:
                result = [item1 - item2 for item1, item2 in zip(non_self_referential_scores_zero_shot, non_self_referential_comparison_scores_zero_shot)]
                bootstrap_result = scipy.stats.bootstrap(method="basic", data=(result,), statistic=scipy.mean)
                data_non_self_referential_comparison["$\\Delta$ Val ZS$^L$"].append(display_entry(sum(result)/len(result), (bootstrap_result.confidence_interval.high - bootstrap_result.confidence_interval.low)/2, 0, only_bold_above_random=False))
            else:
                data_non_self_referential_comparison["$\\Delta$ Val ZS$^L$"].append("-")

            if len(rel_scores_zero_shot) > 1:
                bootstrap_result_rel_zero_shot = scipy.stats.bootstrap(method="basic", data=(rel_scores_zero_shot,), statistic=scipy.mean)
                data["Val ZS$^L$ (R)"].append(display_entry(sum(rel_scores_zero_shot)/len(rel_scores_zero_shot), (bootstrap_result_rel_zero_shot.confidence_interval.high - bootstrap_result_rel_zero_shot.confidence_interval.low)/2, 0.5))
            else:
                data["Val ZS$^L$ (R)"].append("-")

            if len(non_self_referential_rel_scores_zero_shot) > 1:
                result = [item1 - item2 for item1, item2 in zip(non_self_referential_rel_scores_zero_shot, non_self_referential_comparison_rel_scores_zero_shot)]
                bootstrap_result = scipy.stats.bootstrap(method="basic", data=(result,), statistic=scipy.mean)
                data_non_self_referential_comparison["$\\Delta$ Val ZS$^L$ (R)"].append(display_entry(sum(result)/len(result), (bootstrap_result.confidence_interval.high - bootstrap_result.confidence_interval.low)/2, 0, only_bold_above_random=False))
            else:
                data_non_self_referential_comparison["$\\Delta$ Val ZS$^L$ (R)"].append("-")

            if len(scores_cot) > 1:
                bootstrap_result_cot = scipy.stats.bootstrap(method="basic", data=(scores_cot,), statistic=scipy.mean)
                data["Val CoT$^T$"].append(display_entry(sum(scores_cot)/len(scores_cot), (bootstrap_result_cot.confidence_interval.high - bootstrap_result_cot.confidence_interval.low)/2, 0.5))
            else:
                data["Val CoT$^T$"].append("-")

            if len(non_self_referential_scores_cot) > 1:
                result = [item1 - item2 for item1, item2 in zip(non_self_referential_scores_cot, non_self_referential_comparison_scores_cot)]
                bootstrap_result = scipy.stats.bootstrap(method="basic", data=(result,), statistic=scipy.mean)
                data_non_self_referential_comparison["$\\Delta$ Val CoT$^T$"].append(display_entry(sum(result)/len(result), (bootstrap_result.confidence_interval.high - bootstrap_result.confidence_interval.low)/2, 0, only_bold_above_random=False))
            else:
                data_non_self_referential_comparison["$\\Delta$ Val CoT$^T$"].append("-")

        else:
            raise ValueError(f"Unsupported filename in evaluation results: {filename}")
        
    add_dash_for_missing_entries(data)
    add_dash_for_missing_entries(data_non_self_referential_comparison)

    data_df = pd.DataFrame(data)
    data_df["Model"] = data_df["filename"].apply(lambda name: name_to_pretty_name_parameters_chat_order(name)[0])
    data_df["Params"] = data_df["filename"].apply(lambda name: name_to_pretty_name_parameters_chat_order(name)[1])
    data_df["Chat"] = data_df["filename"].apply(lambda name: name_to_pretty_name_parameters_chat_order(name)[2])
    data_df["order"] = data_df["filename"].apply(lambda name: name_to_pretty_name_parameters_chat_order(name)[3])
    data_df = data_df.sort_values(by='order')
    data_df.drop("filename", axis=1, inplace=True)
    data_df.drop("order", axis=1, inplace=True)
    print(data_df.to_latex(index=False).replace("\\begin{tabular}{lllllllll}", "\\begin{tabular}{lrr|rrrrrr}"))
    if non_self_referential_comparison:
        data_non_self_referential_comparison_df = pd.DataFrame(data_non_self_referential_comparison)
        data_non_self_referential_comparison_df["Model"] = data_non_self_referential_comparison_df["filename"].apply(lambda name: name_to_pretty_name_parameters_chat_order(name)[0])
        data_non_self_referential_comparison_df["Params"] = data_non_self_referential_comparison_df["filename"].apply(lambda name: name_to_pretty_name_parameters_chat_order(name)[1])
        data_non_self_referential_comparison_df["Chat"] = data_non_self_referential_comparison_df["filename"].apply(lambda name: name_to_pretty_name_parameters_chat_order(name)[2])
        data_non_self_referential_comparison_df["order"] = data_non_self_referential_comparison_df["filename"].apply(lambda name: name_to_pretty_name_parameters_chat_order(name)[3])
        data_non_self_referential_comparison_df = data_non_self_referential_comparison_df.sort_values(by='order')
        data_non_self_referential_comparison_df.drop("filename", axis=1, inplace=True)
        data_non_self_referential_comparison_df.drop("order", axis=1, inplace=True)
        print(data_non_self_referential_comparison_df.to_latex(index=False).replace("\\begin{tabular}{lllllllll}", "\\begin{tabular}{lrr|rrrrrr}"))
        return data_df, data_non_self_referential_comparison_df
    return data_df

# Initialize an empty dictionary to store tags and their counts
tag_counts_and_ids = {}

# Open the JSON Lines file
with open('i_am_a_strange_dataset.jsonl', 'r') as file:
    for line in file:
        # Parse each line as JSON
        data = json.loads(line)

        # Split the "tag" string by spaces
        tags = data['tag'].split()
        # Update the count for each tag
        for tag in tags:
            if tag in tag_counts_and_ids:
                tag_counts_and_ids[tag][0] += 1
                tag_counts_and_ids[tag][1].append(data['id'])
            else:
                tag_counts_and_ids[tag] = [1, [data['id']]]

# Results for whole dataset.
data_df, data_non_self_referential_comparison_df = print_results()

scores = []
cot_scores = []
for index, row in data_non_self_referential_comparison_df.iterrows():
    for metric in ["$\\Delta$ Gen$^L$", "$\\Delta$ Val ZS$^L$",  "$\\Delta$ Val FS$^L$", "$\\Delta$ Val ZS$^L$ (R)", "$\\Delta$ Val FS$^L$ (R)"]:
        if row[metric] == "-":
            continue
        scores.append(float(row[metric].replace("\\textbf{", "")[:5]))

avg_score = sum(scores)/len(scores)
bootstrap_result = scipy.stats.bootstrap(method="basic", data=(scores,), statistic=scipy.mean)
print("Avg delta score without CoT:", f"{avg_score} $\pm$ {(bootstrap_result.confidence_interval.high - bootstrap_result.confidence_interval.low)}")


# Print graph of parameter count to average score.
params_to_scores = {}
for index, row in data_df.iterrows():
    params = row["Params"].replace("B", "")
    if params == "-":
        continue
    elif params == "8x7":
        params = 8*7
    else:
        params = float(params)
    params = float(params)
    scores = []
    for metric in ["Gen$^L$", "Val ZS$^L$",  "Val FS$^L$", "Val ZS$^L$ (R)", "Val FS$^L$ (R)"]:
        scores.append(float(row[metric].replace("\\textbf{", "")[:5]))
    avg_score = sum(scores)/len(scores)
    params_to_scores[params] = params_to_scores.get(params, [])
    params_to_scores[params].append(avg_score)

points = []
for key, value in params_to_scores.items():
    points.append(f"({key}, {sum(value)/len(value)})")

print("""
\\begin{figure}[ht]
\\centering
\\resizebox{\\columnwidth}{!}{
\\begin{tikzpicture}
\\begin{axis}[
    title={Params vs Avg Logit-based Score},
    xlabel={Params (B)},
    ylabel={Score},
    grid=major,
    xmin=0, xmax=70,
    ymin=52, ymax=56
]
\\addplot[mark=o, blue] coordinates {
""")
for point in points:
    print(point)
print("""
};
\end{axis}
\end{tikzpicture}
}
\caption{A plot of params to average ``I am a Strange Dataset'' score accross all of the logit-based metrics.}
\label{fig:paramstoavgscore}
\end{figure}
""")


# Results for each tag.
tag_and_avg_score = []
tag_and_avg_score_non_self_referential_comparison = []
for key, value in tag_counts_and_ids.items():
    print("\\begin{table*}")
    print("\\centering")
    print("\\resizebox{\\textwidth}{!}{")
    data_df = print_results(value[1], non_self_referential_comparison=False)
    print("}")
    print("\\caption{Results for the " + str(value[0]) + " example pairs with the " + key + " tag. Scores with 95\\% confidence intervals above chance are shown in \\textbf{bold}." + "}")
    print("\\end{table*}")

    scores = []
    cot_scores = []
    for index, row in data_df.iterrows():
        for metric in ["Gen$^L$", "Val ZS$^L$",  "Val FS$^L$", "Val ZS$^L$ (R)", "Val FS$^L$ (R)"]:
            if row[metric] == "-":
                continue
            scores.append(float(row[metric].replace("\\textbf{", "")[:5]))
        if row["Val CoT$^T$"] == "-":
            continue
        cot_scores.append(float(row["Val CoT$^T$"].replace("\\textbf{", "")[:5]))
    avg_score = sum(scores)/len(scores)
    avg_full_score = sum(scores + cot_scores)/len(scores + cot_scores)
    tag_and_avg_score.append((key, avg_score, avg_full_score))

    scores = []
    cot_scores = []
    for index, row in data_non_self_referential_comparison_df.iterrows():
        for metric in ["$\\Delta$ Gen$^L$", "$\\Delta$ Val ZS$^L$",  "$\\Delta$ Val FS$^L$", "$\\Delta$ Val ZS$^L$ (R)", "$\\Delta$ Val FS$^L$ (R)"]:
            if row[metric] == "-":
                continue
            scores.append(float(row[metric].replace("\\textbf{", "")[:5]))
        if row["$\\Delta$ Val CoT$^T$"] == "-":
            continue
        cot_scores.append(float(row["$\\Delta$ Val CoT$^T$"].replace("\\textbf{", "")[:5]))
    avg_score = sum(scores)/len(scores) if len(scores) > 0 else 0
    avg_full_score = sum(scores + cot_scores)/len(scores + cot_scores) if len(scores + cot_scores) > 0 else 0
    tag_and_avg_score_non_self_referential_comparison.append((key, avg_score, avg_full_score))

sorted_tag_and_avg_score = sorted(tag_and_avg_score, key=lambda x: x[1])
sorted_tag_and_avg_score_incl_cot = sorted(tag_and_avg_score, key=lambda x: x[2])

sorted_tag_and_avg_score_non_self_referential_comparison = sorted(tag_and_avg_score_non_self_referential_comparison, key=lambda x: x[1])
sorted_tag_and_avg_score_incl_cot_non_self_referential_comparison = sorted(tag_and_avg_score_non_self_referential_comparison, key=lambda x: x[2])

print()
print("Sorted average tag scores not including CoT:", [(obj[0], obj[1]) for obj in sorted_tag_and_avg_score])
print()

points = []
for key, value in params_to_scores.items():
    points.append((key, sum(value)/len(value)))


# Now compute the mechanical turk results for the dataset:
df_batch_1 = pd.read_csv("mechanical_turk/results/Batch_5171266_batch_results.csv")
df_batch_2 = pd.read_csv("mechanical_turk/results/Batch_5171265_batch_results.csv")
id_to_human_judgements = {}
concatenated_df = pd.concat([df_batch_1, df_batch_2], ignore_index=True)

for index, row in concatenated_df.iterrows():
    id_to_human_judgements[row["Input.id"]] = id_to_human_judgements.get(row["Input.id"], {"true_continuation_judgements": [], "false_continuation_judgements": []})
    if row["Input.gold_label"]:
        if row["Answer.label.label"]:
            judgement = 1
        else:
            judgement = 0
        id_to_human_judgements[row["Input.id"]]["true_continuation_judgements"].append(judgement)
    else:
        if row["Answer.label.label"]:
            judgement = 1
        else:
            judgement = 0
        id_to_human_judgements[row["Input.id"]]["false_continuation_judgements"].append(judgement)

human_losses = {"l_true_statement_true_answer": [], "l_true_statement_false_answer": [], "l_false_statement_true_answer": [], "l_false_statement_false_answer": []}
for key, value in id_to_human_judgements.items():

    # Get probabilities from humans for each condition.
    p_true_statement_true_answer = sum(value["true_continuation_judgements"])/len(value["true_continuation_judgements"])
    p_true_statement_false_answer = 1 - p_true_statement_true_answer

    p_false_statement_true_answer = sum(value["false_continuation_judgements"])/len(value["false_continuation_judgements"])
    p_false_statement_false_answer = 1 - p_false_statement_true_answer

    # Convert to negative log likelihood (loss) and add to the human loss list.
    def nll(x):
        assert x >= 0 and x <= 1
        if x == 0:
            return float("inf")
        else:
            return abs(math.log(x))

    human_losses["l_true_statement_true_answer"].append(nll(p_true_statement_true_answer))
    human_losses["l_true_statement_false_answer"].append(nll(p_true_statement_false_answer))
    human_losses["l_false_statement_true_answer"].append(nll(p_false_statement_true_answer))
    human_losses["l_false_statement_false_answer"].append(nll(p_false_statement_false_answer))

df_human_losses = pd.DataFrame(human_losses)
scores_val = []
scores_rel_val = []

for index, row in df_human_losses.iterrows():
    val_score("l_true_statement_true_answer", "l_true_statement_false_answer", "l_false_statement_true_answer", "l_false_statement_false_answer", scores_val, row)
    rel_val_score("l_true_statement_true_answer", "l_true_statement_false_answer", "l_false_statement_true_answer", "l_false_statement_false_answer", scores_rel_val, row)

print("Human Val")
bootstrap_result_val = scipy.stats.bootstrap(method="basic", data=(scores_val,), statistic=scipy.mean)
print(display_entry(sum(scores_val)/len(scores_val), (bootstrap_result_val.confidence_interval.high - bootstrap_result_val.confidence_interval.low)/2, 0.5))

print("Human Rel Val")
bootstrap_result_rel_val = scipy.stats.bootstrap(method="basic", data=(scores_rel_val,), statistic=scipy.mean)
print(display_entry(sum(scores_rel_val)/len(scores_rel_val), (bootstrap_result_rel_val.confidence_interval.high - bootstrap_result_rel_val.confidence_interval.low)/2, 0.5))

import pandas as pd
from utils import write_conversation_to_notebook
from user_requests import enriched_user_requests_ver0, enriched_user_requests_ver1, enriched_user_requests_ver2
import openai
from datasets_info import medium_dataset_info, ds_salary_info
import time

openai.api_key = "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"
pre = [{"role": "system", "content": "Produce only Python code in your response as you are a code generator. \
                                                    If you need to include any notes or explanations in natural language, \
                                                    mark them with the '#' symbol."},
                    {"role": "user", "content": "load data.csv file using pandas"},
                    {"role": "assistant", "content": "import pandas as pd\n"
                                                     "# Load the CSV file into a pandas dataframe\n"
                                                     "df = pd.read_csv('data.csv')\n"
                                                     "# Print the first 5 rows of the dataframe to verify that it was loaded correctly\n"
                                                     "print(df.head())"},
                    {"role": "user", "content": "drop the 'example' column"},
                    {"role": "assistant", "content": "# Drop example from train\n"
                                                     "train = train.drop(['example'], axis=1)\n"
                                                     "# Drop example from test\n"
                                                     "test = test.drop(['example'], axis=1)\n"},
                    {"role": "user", "content": "add 10% to the 'price' column"},
                    {"role": "assistant", "content": "# add 10% to 'price' column\n"
                                                     "train['date'] *= 1.1\n"
                                                     "test['date'] *= 1.1\n"},
                    {"role": "user", "content": "instantiate a binary classification model"},
                    {"role": "assistant", "content": "# load LogisticRegression binary classification model\n"
                                                     "from sklearn.linear_model import LogisticRegression\n"
                                                     "model = LogisticRegression()"}]
conv_intro_to_keep = len(pre)


def truncate_context(conversation, data_info=None):
    if len(conversation) > 25:
        if data_info:
            return conversation[:conv_intro_to_keep+2] + conversation[conv_intro_to_keep+7:]
        else:
            return conversation[:conv_intro_to_keep] + conversation[conv_intro_to_keep+5:]
    else:
        return conversation


def generate_ds_code(requests, dataset_name, k, data_info=None):
    conversation = pre
    if data_info:
        conversation.append({"role": "assistant", "content": f"The dataset_name is: {dataset_name}, "
                                                             f"use it in the following requests when required"})
        conversation.append({"role": "assistant", "content": "The dataset information is: " + data_info})
    generated_code = []
    for i, request in enumerate(requests):
        inference_not_done = True
        while inference_not_done:
            try:
                time.sleep(20)
                conversation.append({"role": "user", "content": request})
                conversation = truncate_context(conversation, data_info)
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    temperature=0,
                    messages=conversation
                )
                conversation.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
                generated_code.append(response['choices'][0]['message']['content'])
                print(f"Run {k}: Request {i} completed")
                inference_not_done = False
            except Exception as e:
                print(f"Waiting 2 minutes")
                print(f"Error was: {e}")
                time.sleep(120)
    return generated_code


def write_code_to_file(generated_code, save_path, k):
    with open(save_path+str(k)+'.py', "w") as out:
        for code_block in generated_code:
            out.write('try:\n\t')
            out.write('\n\t'.join(code_block.split("\n"))+"\n")
            out.write('except:\n\tpass\n')


def run_code(path):
    try:
        imported_py_file = __import__(path)
    except:
        return None
    try:
        return imported_py_file.mse
    except:
        pass
    try:
        return imported_py_file.accuracy
    except:
        return None


def main(requests, dataset_name, data_info, save_path, k_trials):
    scores = []
    for k in range(k_trials):
        gen_code = generate_ds_code(requests, dataset_name, k, data_info)
        write_code_to_file(gen_code, save_path, k)
        scores.append(run_code(save_path+str(k)))
        print(scores)
    return scores


if __name__ == "__main__":
    print(main(enriched_user_requests_ver2, 'medium', medium_dataset_info, "medium", 10))



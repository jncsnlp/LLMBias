import pandas as pd
import argparse
import json

#shell 
# python process_sst2.py --file_name train.jsonl --data_path ./data/sst2 --instruct "Give the emotional polarity of the following sentences." --labels "['Positive', 'Negative']"
def parse_args():

    parser = argparse.ArgumentParser(description="Coverting the classification dataset.")
    parser.add_argument('--file_name', nargs='?', default='train.csv',\
                        help='File name for the dataset.')
    parser.add_argument('--data_path', nargs='?', default='./data',
                        help='File path for the dataset.')
    parser.add_argument('--instruct', nargs='?', default='Is the following sententce sentiment positive or negative?',
                        help='Instruction for the dataset.')
    parser.add_argument('--labels', nargs='?', default='[\'Positive\', \'Negative\']',
                        help='Labels for the dataset.')
    return parser.parse_args()


if __name__=="__main__":
    args=parse_args()
    print('Arguments:\n{}'.format(args))

    FILE_PATH = args.data_path + '/' + args.file_name
    DES_PATH = args.data_path + '/converted_' + args.file_name.split('.')[0] + '.json'

    
    origin_data = pd.read_json(FILE_PATH, lines=True)

    mapping_labels = eval(args.labels)

    converted_data = []

    for row in origin_data.itertuples():
        elem = {"instruction": args.instruct, "input": row.sentence, "output": "{}".format(mapping_labels[int(row.label)])}
        converted_data.append(elem)

    with open(DES_PATH, 'w') as f:
        json.dump(converted_data, f, indent = 4)

    print("The converted data has been saved to {}!".format(DES_PATH))
import os
import datasets
import random

def make_map_fn(split, source=None):
        def process_fn(example, idx):
            if source is None:
                data_source = example.pop("source")
            else:
                data_source = source
            question = example.pop("prompt")
            solution = example.pop("answer")
            
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": str(solution)},
                "extra_info": {
                    "split": split,
                    "index": f"{data_source}-{idx}",
                },
            }
            return data

        return process_fn

if __name__ == '__main__':
    
    data_dir = "/home/test/test06/hbx/JustRL/train/data"

    # data_source_list = ['AIME25', 'AIME24', 'AMC23', 'MATH-500', 'Minerva',
    #                     'Olympiad-Bench', 'GPQA', 'MedQA', 'MedXpertQA-Text', 'MMLU-Pro']
    # for data_source in data_source_list:
    #     test_dataset = datasets.load_dataset("json", data_files=os.path.join(data_source, 'test.json'), split='train')
    #     test_dataset = test_dataset.map(function=make_map_fn("test", data_source), with_indices=True)
    #     test_dataset.to_parquet(os.path.join(data_source, 'test.parquet'))

    # data_source_list = ['MATH', 'MATH-8k', 'DAPO', 'DeepScaler', 'ORZ']
    # for data_source in data_source_list:
    #     train_dataset = datasets.load_dataset("json", data_files=os.path.join(data_source, 'train.json'), split='train')
    #     train_dataset = train_dataset.map(function=make_map_fn("train", data_source), with_indices=True)
    #     train_dataset.to_parquet(os.path.join(data_source, 'train.parquet'))

    data_source_list = ['AIME25', 'AIME24', 'AMC23']
    for data_source in data_source_list:
        test_dataset = datasets.load_dataset("json", data_files=os.path.join(data_dir, data_source, 'test.json'), split='train')
        test_dataset = test_dataset.map(function=make_map_fn("test", data_source), with_indices=True)
        test_dataset.to_parquet(os.path.join(data_dir, data_source, 'test.parquet'))

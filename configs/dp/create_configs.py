import json

model = '2.7B'
batch = 128
f = open(f'/home/lklab/knowledge-unlearning/configs/dp/template.json')
data = json.load(f)
print(data)

for i in range(0, 5):
    data['wandb_run_name'] = f'DP-0.2-{model}_{i}'
    data['model_name_or_path'] = f"EleutherAI/gpt-neo-{model}"
    data['train_set'] = f'data/main/lm_extraction_32_{i}.csv'
    data['valid_sets'][0] = f'data/main/lm_extraction_32_{i}.csv'
    with open(f'/home/lklab/knowledge-unlearning/configs/dp/{model}_{i}.json', 'w') as fp:
        json.dump(data, fp, indent=4)
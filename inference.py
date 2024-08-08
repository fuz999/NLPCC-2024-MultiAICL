# Import relevant libraries and dependencies
import os
from pathlib import Path
import argparse
import json
import torch
from transformers import (
    GPTJForCausalLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed, AutoConfig,
)
from tqdm import tqdm

FS_EOS_TOKEN = '\n###\n'

# Model dictionary
MODEL_DICT = {
    'gpt2': 'gpt2',
    'gpt2-medium': 'gpt2_medium', 
    'gpt2-large': 'gpt2_large', 
    'gpt2-xl': 'gpt2_xl', 
    'EleutherAI/gpt-neo-1.3B': 'gptneo_1.3B',
    'EleutherAI/gpt-neo-2.7B': 'gptneo_2.7B',
    'EleutherAI/gpt-j-6B': 'gptj_6B',
}

# Dataset dictionary
DATASET_DICT ={
    'yelp': {
        'from': ['positive', 'negative'], 
        'from_to': {
            'positive': 'negative',
            'negative': 'positive',
            },
        'full_style_description': None,
        'size': 500,
        'examples': [
            ('positive', 'this place is amazing!','negative', 'this place is awful!'),
            ('positive', 'good drinks,and good company.', 'negative', 'very unhappy with the usa company.'),
            ('positive', 'definitely will buy another pair of socks from this store--they have the best socks ever','negative','definitely will NOT buy another pair of socks from this store--they have the worst socks ever'),
            ('negative','my wife and i were disappointed by the quality of the service--also, the food was pretty tasteless','positive','my wife and i were impressive by the quality of the service--also, the food was pretty delicious'),
            ('positive', 'i loved their black tea and hot chocolate selections!', 'negative','i hated their black tea and hot chocolate selections!'),
            ('negative', 'the drinks were weak and pour.', 'positive', 'the drinks were affordable and a good pour.'),
            ('positive', 'the chicken chimi i had was absolutely fantastic!', 'negative','the chicken chimi i had was crappy.'),]
    },
    'amazon': {
        'from': ['positive', 'negative'], 
        'from_to': {
            'positive': 'negative',
            'negative': 'positive',
            },
        'full_style_description': None,
        'size': 500,
        'examples' : [
            ('positive', 'very small but it works great in the car.', 'negative', 'very small and it works terribly in the car.'),
            ('positive', 'i really loved it and will use it alot.', 'negative', 'i really disliked it and will not use it again.'),
            ('positive', 'it gets the job done and for the price you can t beat it.', 'negative', 'it does not work well and it was expensive.'),
            ('negative', 'i will never buy anything from this brand again.', 'positive', 'i will buy from this brand again.'),
            ('negative', 'i have to say that i am not a fan of this item.', 'positive', 'i have to say that i am a big fan of this item'),
            ('negative', 'i bought the charger in june and it was broken by august.', 'positive', 'i bought the charger in june and it still works well.'),
        ],
    },
    'gyafc': {
        'from': ['informal', 'formal'], 
        'from_to': {
            'informal': 'formal',
            'formal': 'informal',
            },
        'full_style_description': None,
        'size': 500,
        'examples' : [
            ('informal', 'sorry but donnt know if i can do this alone.', 'formal', 'I am sorry, but I don\'t know if I can do this alone.'),
            ('formal', 'i am going to ask him to come to the concert with me, and i hope he accepts my invitation.', 'informal', 'gonna ask him to come to the concert with me and hope he says yes :)'),
            ('informal', 'that sucks man but u gotta move on', 'formal', 'that is unfortunate, but you need to move on'),
            ('formal', 'and i am sorry that you and your girlfriend broke up last week.', 'informal', 'and im sorry that u and ur girlfriend broke up last week...'),

        ]
    },
    'jfleg': {
        'from': ['ungrammatical'], 
        'from_to': {
            'ungrammatical': 'grammatical',
            },
        'full_style_description': None,
        'size': 747,
        'examples': [
            ('ungrammatical', 'There are several reason.', 'grammatical', 'There are several reasons.'),
            ('ungrammatical', 'To my surprize nothing happened.', 'grammatical', 'To my surprise, nothing happened.'),
            ('ungrammatical', 'This is important thing.', 'grammatical', 'This is an important thing.'),
            ('ungrammatical', 'Water is needed for alive.', 'grammatical', 'Water is necessary to live.'),
            ('ungrammatical', 'And young people spend time more ther lifestile.', 'grammatical', 'And young people spend more time on their lifestyles.'),
            ('ungrammatical', 'Both of these men have dealed with situations in an unconventional manner and the results are with everyone to see.', 'grammatical', 'Both of these men have dealt with situations in an unconventional manner and the results are plain to see.'),
        ],
    },
     'shakespeare': {
        'from': ['Shakespearean',], 
        'from_to': {
            'Shakespearean': 'modern',
            },
        'full_style_description': None,
        'size': 599,
        'examples': [
            ('Shakespearean', 'what hast thou there?', 'modern', 'what have you got there?'),
            ('Shakespearean', 'what say\'st thou, my dear nurse?', 'modern', 'what did you say, my dear nurse?'),
            ('Shakespearean', 'and how doth she?', 'modern', 'and how is she doing?'),
            ('Shakespearean', 'talk not to me, for i\'ll not speak a word.', 'modern', 'don\'t talk to me, because i won\'t answer you.'),
            ('Shakespearean', 'pardon, i beseech you!', 'modern', 'forgive me, i beg you!'),
            ('Shakespearean', 'all men call thee fickle.', 'modern', 'all men say you are changeable.'),
            ('Shakespearean', 'and sayest thou yet that exile is not death?', 'modern','and you still say that exile is not death!'),
        ],
    },
    'symbolic_manipulation': {
        'from': ['symbolic',], 
        'from_to': {
            'symbolic': 'English',
            },
        'full_style_description': None,
        'size': 1000,
        'examples': [
            ('symbolic', 'apple > seven', 'English', 'apple is greater than seven'),
            ('symbolic', 'tiger < robin', 'English', 'tiger is less than robin'),
            ('symbolic', 'cyan > green', 'English', 'cyan is greater than green'),
            ('symbolic', 'apple < dog', 'English', 'apple is less than dog'),
            ('symbolic', 'yellow > apple', 'English', 'yellow is greater than apple'),
            ('symbolic', 'brown < five', 'English', 'brown is less than five'),
        ],
    },
}


# Prompt setting choices
PROMPT_SETTING_CHOICES = [
    'vanilla', 
    'contrastive', 
    'negation_v1',
    'negation_v2',
]


# Clean the output
def clean_output(output, delim_left, delim_right, start_idx=2):
    try:
        generated_output = output.split(delim_left)[start_idx].split(delim_right)[0]
    except:
        print('***')
        print(output)
        print('***')
        import pdb
        pdb.set_trace()
    generated_output = generated_output.replace('\n', ' ')
    return generated_output


# Write the sentence
def write_sentence(setting, delim_left, delim_right, orig_style, opp_style, orig_text, rewritten_text=None, full_style_description=None):
    if full_style_description is not None:
        orig_style = full_style_description[orig_style]
        opp_style = full_style_description[opp_style]

    if setting == 'contrastive':
       sentence = f'{orig_style}: {delim_left}{orig_text}{delim_right}  {opp_style}: {delim_left}'


    if rewritten_text is not None:
        sentence = f'{sentence}{rewritten_text}{delim_right}'
    return sentence

# Create examplars (for the few-shot setting)
def create_examplers(examples, setting, delim_left, delim_right, full_style_description):
    prefix = ''
    for example in examples:
        # ('negative', 'this place is awful!', 'positive', 'this place is amazing!'),
        orig_style, orig_text, opp_style, rewritten_text = example
        add_text = write_sentence(setting, delim_left, delim_right, orig_style, opp_style, orig_text, rewritten_text, full_style_description) #调用上面那个生成句子函数
        prefix += f'{add_text}{FS_EOS_TOKEN}'
    return prefix
    

# Perform inference
def inference(
    model,
    tokenizer,
    data,
    setting,
    k_samples,
    num_examplars,
    prefix,
    orig_style,
    opp_style,
    full_style_description,
    max_examples,
    save_path,
    max_model_token_length,
    delim_left,
    delim_right,
    device,
):
    # Reset the output file
    with open(save_path, 'w') as f:
        pass

    # Loop
    pbar = tqdm(data, desc='Generating examples')
    for i, sample in enumerate(pbar):
        if i == max_examples:
            break

        # Data
        text = sample[:-1]
        
        # Prompt

        prompt = write_sentence(setting, delim_left, delim_right, orig_style, opp_style, text, None)
        prompt = f'{prefix}{prompt}'
        print(prompt)
        print('1')
        # Feed the input and get the output
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        generated_ids = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=max_model_token_length, num_return_sequences=k_samples)

        outputs = []
        generated_outputs = []

        for k in range(k_samples):
            output = tokenizer.decode(generated_ids[k])
            # Clean output (such as: replace '\n' with ' ')
            generated_output = clean_output(output, delim_left, delim_right, start_idx=(num_examplars+1)*2)
            outputs.append(output)
            generated_outputs.append(generated_output)


        with open(save_path, 'a') as f:
            print(json.dumps({
                'text': text,
                'prompt': prompt,
                'output': outputs,
                'generated_output': generated_outputs
            }), file=f)


        if i % 10 == 0:
            text_clean = text.replace("\n", " ")
            pbar.write(f'[{i}] Text:       {text_clean}')
            pbar.write(f'[{i}] Generated:  {generated_outputs[0]}')


def main():
    parser = argparse.ArgumentParser()

    # Parser arguments
    parser.add_argument('--model', type=str, default='gpt2-xl') #choices=MODEL_DICT.keys()
    parser.add_argument('--tokenizer', type=str, default=None) #choices=MODEL_DICT.keys()
    parser.add_argument('--datasets', type=str, default='yelp', choices=DATASET_DICT.keys())
    parser.add_argument('--clean_data', action='store_true')
    parser.add_argument('--setting', type=str, default='contrastive', choices=PROMPT_SETTING_CHOICES)
    parser.add_argument('--k_samples', type=int, default=1)
    parser.add_argument('--num_examplars', type=int, default=0)
    parser.add_argument('--results_dir', type=str, default=None)
    args = parser.parse_args()

    # Sanity checkpoints
    assert (args.k_samples > 0)
    assert (args.num_examplars >= 0)

    # Axes of variation
    setting = args.setting
    k_samples = args.k_samples
    num_examplars = args.num_examplars
    model_name = args.model
    tokenizer_name = args.tokenizer if args.tokenizer else model_name


    model_name="MultiAICLmodel.pt"
    config_path = "config.json"
    tokenizer_name = 'gpt2-xl'

    dataset_name = args.datasets

    dataset_info = DATASET_DICT[dataset_name]
    full_style_description = dataset_info['full_style_description']


    delim_left, delim_right = ('[', ']')
    
    # Set seed for reproducibility,
    set_seed(1729)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): print("successful")
    else: print("fail")


    # Model
    print('Loading model...')
    # 加载配置
    config = AutoConfig.from_pretrained(config_path)
    if model_name == 'EleutherAI/gpt-j-6B':
        model = GPTJForCausalLM.from_pretrained(
            model_name,
            revision="float16", 
            low_cpu_mem_usage=True,
        ).eval().to(device)
    else:
         model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name,config=config).eval().to(device)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print(f'Loaded model. Parameters: {sum(p.numel() for p in model.parameters()):_}')

    # Parameters
    max_model_token_length = 256 if num_examplars == 0 else 512
    max_examples = dataset_info['size']

    # Paths
    dataset_name = f'{dataset_name}_clean' if args.clean_data else dataset_name
    data_dir = f'datasets1/{dataset_name}'

    # Output directory
    '''
    results_dir = f'outputs/{dataset_name}/{setting}/{model_acronym}/{num_examplars}_shot_{k_samples}_samples' if (args.results_dir is None) else args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    '''
    results_dir = f'outputs/{dataset_name}/{setting}/gpt2-xl/{num_examplars}_shot_{k_samples}_samples' if (
                args.results_dir is None) else args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    prefix = create_examplers(dataset_info['examples'][:num_examplars], setting, delim_left, delim_right, full_style_description) if num_examplars > 0 else ''
    print(prefix)
    print('2')


    for orig_style in dataset_info['from']:

        print('Loading the datasets1...')
        dataset_path = os.path.join(data_dir, f'gt_{orig_style}_input.txt')
        data = open(dataset_path, 'r').readlines()
        print('*** The datasets1 is loaded.')
        
        # Target style
        opp_style = dataset_info['from_to'][orig_style]


        save_path = os.path.join(results_dir, f'gpt2-xl_{orig_style}_{args.delimiter}.jsonl')
        if Path(save_path).is_file() and len(Path(save_path).read_text().splitlines()) == 500:
            print(f'Skipping because it already contains 500 lines: {save_path}')
            continue
        inference(
            model,
            tokenizer,
            data,
            setting,
            k_samples,
            num_examplars,
            prefix,
            orig_style,
            opp_style,
            full_style_description,
            max_examples,
            save_path,
            max_model_token_length,
            delim_left,
            delim_right,
            device
        )


if __name__ == "__main__":
    main()

from transformers import AutoTokenizer 
from transformers import AutoModelForCausalLM
from datasets import load_dataset
import torch
import torch.nn as nn 
from peft import PeftModel, PeftConfig 
from tqdm import tqdm
import fnmatch

def evaluate_ppl(dataset_name, model, tokenizer, ctx_length):
    # max_length = model.seqlen 
    model_seqlen = ctx_length
    max_length = ctx_length
    stride = ctx_length

    if dataset_name == "wikitext":
        test = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test')
        encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt").to('cuda')
        seq_len = encodings.input_ids.size(1)
    elif dataset_name == "ptb":
        testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')
        encodings = tokenizer(" ".join(testdata['sentence']), return_tensors='pt').to('cuda')
        seq_len = encodings.input_ids.size(1)
    elif dataset_name == "c4":
        config_name = "default-c7bc8b0aefc5e48f"
        valdata = load_dataset(
            'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
        )
        # valdata = load_dataset('allenai/c4', config_name, split='validation')
        encodings = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt').to('cuda')
        # encodings = encodings.input_ids[:, :(256 * model.seqlen)]
        seq_len = 256 * model_seqlen

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to('cuda')
        target_ids = input_ids.clone().to('cuda')
        target_ids[:, :-trg_len] = -100
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item() 


def eval_llm(model_name_or_path, tokenizer, task_list=["lambada","hellaswag","piqa","winogrande"], num_fewshot=0):
# def eval_llm(model_name_or_path, tokenizer, task_list=["cimec/lambada","Rowan/hellaswag","ybisk/piqa","allenai/winogrande"], num_fewshot=0):
    # LAMBADA, HellaSwag, PIQA, and WinoGrande Datasets
    # piqa,hellaswag,winogrande,lambada,openbookqa
    # "lambada","hellaswag","piqa","winogrande","arc_challenge","arc_easy","openbookqa"
    from lm_eval import evaluator
    from lm_eval.tasks import TaskManager
    def pattern_match(patterns, source_list):
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)
    
    task_manager = TaskManager()
    # print("All tasks:", task_manager.all_tasks)
    task_names = pattern_match(task_list, task_manager.all_tasks)
    # task_names = pattern_match(task_list, tasks._all_tasks)

    results = evaluator.simple_evaluate(
        model="hf-auto",
        # model='/root/autodl-tmp/watermark_project/models/facebook/opt-30b-inserted-by-randommark',
        # model_args=f"pretrained={args.model},trust_remote_code=True",
        model_args=f"pretrained={args.model}",
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=32, 
        device='cuda',
        # device=None, 
        use_cache=None,
        limit=None,
        log_samples=False,
        check_integrity=False,
        task_manager=task_manager,
    )
    return results['results']  

def main(args):

    if args.eval_zero_shot:
        task_list_dict = {0: ["lambada_standard","hellaswag","piqa","winogrande"]}
        accelerate=False 
        for num_shot in [0]:
            task_list = task_list_dict[num_shot]
            print("zero_shot evaluation results")
            results = eval_llm(args.model, None, task_list, num_shot)
            
            # 将结果写入到文本文件中
            format_results = {}
            avg_acc = 0.0
            for key, value in results.items():
                acc = value.get('acc,none', -1)
                avg_acc += acc
                format_results[key] = acc
            format_results['avg_acc'] = avg_acc / len(results)
            
            result_file = args.model + "-zero_shot_results.txt"
            with open(result_file, 'a+') as file:
                file.write(f"{args.model}\n")
                for key,value in format_results.items():
                    file.write(f"{key}\t{value}\n")

    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
                    args.model,
                    # low_cpu_mem_usage=True,
                    torch_dtype=torch.float16,
                    # load_in_8bit=True if "8bit" in args.model else None,
                    device_map="auto",
    )

    model.eval()
    result_file = args.model + "-perplexity_results.txt"
    with open(result_file, "a+") as file:
        file.write(f"{args.model}\n")
        ppl = evaluate_ppl(args.dataset_name, model, tokenizer, args.ctx_length)
        file.write(f"Perplexity on wikitext: {ppl}\n")
        

    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str, default="../models/facebook/opt-1.3b-8bit"
    )
    parser.add_argument(
        '--dataset_name', type=str, default="wikitext"  
    )
    parser.add_argument(
        '--ctx_length', type=int, default=2048 
    )
    parser.add_argument("--eval_zero_shot", type=bool, default=True)

    args = parser.parse_args()
    main(args)
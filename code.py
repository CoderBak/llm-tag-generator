import os
import torch
import vllm
import random
import re
import numpy as np
import json
from transformers import AutoTokenizer
from datetime import datetime

class Model():
    def __init__(self, model_path, **model_args):
        """
        This function loads the model following model_args.
        Please refer to vllm documentation to know the list of arguments.
        You can set: tokenizer, tensor_parallel_size, dtype, quantization,
        gpu_memory_utilization, enforce_eager, etc.
        """
        self.seed = int(os.environ["PYTHONHASHSEED"])
        model_args["seed"] = self.seed
        if "tensor_parallel_size" not in model_args:
            model_args["tensor_parallel_size"] = torch.cuda.device_count()
        self.model_path = model_path
        self.llm = vllm.LLM(model=model_path, trust_remote_code=True, **model_args)


    def set_sampling_args(self, **sampling_args):
        """
        This function sets the sampling params. It must be called before evaluation.
        You can set: n, best_of, presence_penalty, frequency_penalty, repetition_penalty, temperature,
        top_p, top_k, min_p, stop, ignore_eos, max_tokens, min_tokens
        """
        self.sampling_args = sampling_args
        self.sampling_args["seed"] = self.seed
        self.sampling_params = vllm.SamplingParams(**sampling_args)


    @staticmethod
    def postprocess(base, prompt):
        result = base.replace("%%Question%%", prompt)
        matches = re.findall(r'%%(.*?)%%', result)

        for match in matches:
            match = match.strip()
            if os.path.exists(match):
                try:
                    with open(match, 'r', encoding='utf-8') as file:
                        file_content = file.read()
                    result = result.replace(f"%%{match}%%", file_content)
                except Exception as e:
                    print(f"Error reading file {match}: {e}")
            else:
                print(f"File {match} not found.")

        return result


    def generate(self, inputs, vis=False):
        # This function generates the inputs based on the sampling args.
        assert(hasattr(self, 'sampling_params'))

        new_prompts = []

        for idx, prompt in enumerate(inputs):
            new_elem = []
            for elem in self.pattern:
                new_elem.append({"role": elem["role"], "content": self.postprocess(elem["content"], prompt)})
            
            if vis and idx == 0:
                print(new_elem)

            new_prompts.append(self.tokenizer.apply_chat_template(
                new_elem,
                tokenize=False,
                add_generation_prompt=True
            ))

        outputs = self.llm.generate(new_prompts, sampling_params=self.sampling_params)
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        return [[output.outputs[_].text for _ in range(len(output.outputs))] for output in outputs]


    def evaluation(self, inputs, NUM=10, NUM_CLUSTER=15, temperature=0.95, max_tokens=4096, pattern=[]):
        # Inputs should be a list of filename: filecontent
        # This function evaluates the model on a specific dataset.
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.pattern = pattern
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True, trust_remote_code=True)
        names = [_[0] for _ in inputs.items()]
        inputs = [_[1] for _ in inputs.items()]
        self.set_sampling_args(temperature=temperature, max_tokens=max_tokens, n=10)

        to_vote = []
        temp = self.sampling_args["temperature"]
        now = datetime.now()
        time_str = now.strftime("%m.%d-%H:%M")
        base_folder = f"./llm_debug"
        if not os.path.exists(base_folder):
            os.mkdir(base_folder)
        output_place = base_folder + f"/{time_str}_" + self.model_path.split("models")[1].split("/")[-1] + f"_{temp}"
        if not os.path.exists(output_place):
            os.mkdir(output_place)
        
        splitted_contents = []
        for idx, input in enumerate(inputs):
            contents = self.split_text(input, NUM)
            splitted_contents.extend(contents)
        all_results = self.generate(splitted_contents)

        for idx in range(len(inputs)):
            cur_results = all_results[(2 * NUM - 1) * idx: (2 * NUM - 1) * (idx + 1)]
            to_vote.append(cur_results)

        # 对 to_vote 做打分
        best = []
        self.set_sampling_args(temperature=temp, max_tokens=max_tokens, n=1)
        self.pattern = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "请你从下面十个以<summary></summary>包裹的总结中，选出标题总结的最好的一种，并包裹在<best></best>内输出给我。\n\n%%Question%%"}
        ]

        vote_prompts = []
        for cur_split in range(len(inputs)):
            best.append([])
            for idx, result in enumerate(to_vote[cur_split]):
                # 一个 result 有 10 个备选结果
                input = ""
                for k in range(10):
                    input += f"\n<summary>\n{result[k]}\n<\summary>\n"
                vote_prompts.append(input)

        vote_results = self.generate(vote_prompts)
        for cur_split in range(len(inputs)):
            cur_vote_results = vote_results[(2 * NUM - 1) * cur_split: (2 * NUM - 1) * (cur_split + 1)]
            for idx in range(2 * NUM - 1):
                vote = cur_vote_results[idx][0]
                with open(f"{output_place}/{names[cur_split]}.txt", "a", encoding="utf-8") as f:
                    f.write(f"\n第 {idx} 段的最佳总结\n")
                    f.write(vote)
                best[cur_split].append(vote)

        stage = []
        original_keywords = []
        for idx, result in enumerate(best):
            cur_stage = []
            original_keywords.append({})
            for k in range(NUM * 2 - 1):
                text = result[k]
                pattern = r"#\s+(\w+)"
                matches = re.findall(pattern, text)
                for word in matches:
                    original_keywords[idx][word] = k
                cur_stage += matches
                with open(f"{output_place}/phrases_{names[idx]}.txt", "a", encoding="utf-8") as f:
                    f.write(f"第 {k} 段的内容：")
                    f.write(contents[k])
                    f.write("\n关键词：")
                    f.write(", ".join(matches))
                    f.write("\n\n\n")
            stage.append(cur_stage)

        for idx in range(len(best)):
            with open(f"{output_place}/phrases_{names[idx]}.txt", "a", encoding="utf-8") as f:
                f.write("#####################################################################\n\n")
                f.write(f"所有关键词：\n{stage[idx].__str__()}\n\n")
        
            with open(f"{output_place}/original_keywords_{names[idx]}.txt", "w", encoding="utf-8") as f:
                f.write(original_keywords[idx].__str__())
        
        self.pattern = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "你是一名资深的社会学家。下面我将给你一些词语，你需要保留那些社会学信息量高的词语，而删去那些社会学信息量低的词语，例如“会议内容”“研究方法”这些没有代表性的词语。直接使用逗号连接你所选择的词语并输出即可。同时，请你帮助我进行词语的精简，对于含义相近或有多个字重复的词语，你只需要保留一个，从而保证词语的高信息密度和高质量。\n%%Question%%"},
        ]
        clean_prompts = [', '.join(original_keywords[idx].keys()) for idx in range(len(best))]
        clean_results = self.generate(clean_prompts)
        for idx in range(len(best)):
            with open(f"{output_place}/phrases_{names[idx]}.txt", "a", encoding="utf-8") as f:
                f.write("#####################################################################\n\n")
                f.write(f"清洗后关键词：\n{clean_results[idx][0]}\n\n")


        self.pattern = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"请你对下面的关键词进行聚类，划分为 {NUM_CLUSTER} 类。请严格仿照下面的格式：\n1. 法律与制度：\n- 法律指导\n- 规章制度\n- 法律修订\n\n请将你的回答放在<output></output>中。\n\n%%Question%%"}
        ]
        self.set_sampling_args(temperature=temp, max_tokens=max_tokens, n=1)
        cluster_prompts = [clean_results[idx][0].strip() for idx in range(len(best))]
        #cluster_prompts = [', '.join(stage[idx]) for idx in range(len(best))]
        cluster_results = self.generate(cluster_prompts)

        for idx in range(len(best)):
            result = cluster_results[idx][0]
            with open(f"{output_place}/phrases_{names[idx]}.txt", "a", encoding="utf-8") as f:
                f.write(f"\nLLM Cluster: \n{result}\n\n\n")

            def create_keyword_mapping(text):
                mapping = {}
                current_category = None
                
                # 按行分割文本
                lines = text.strip().split('\n')
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    if line[0].isdigit():  # 主类别行 (例如 "1. 信访与矛盾处理：")
                        current_category = line.split('.')[1].split('：')[0].strip()
                    elif line.startswith('-'):  # 关键词行
                        keyword = line[1:].strip()  # 移除 '-' 并清理空白
                        if current_category:
                            mapping[keyword] = current_category
                            
                return mapping

            mapping = create_keyword_mapping(result)
            print(mapping)
            extracted = {}

            for keyword in mapping.keys():
                cluster_result = mapping[keyword]
                if keyword.strip() in original_keywords[idx]:
                    extracted[keyword] = (cluster_result, original_keywords[idx][keyword.strip()])
            
            with open(f"{output_place}/result_{names[idx]}.txt", "w", encoding="utf-8") as f:
                f.write(f"{extracted.__str__()}\n")
            

    @staticmethod
    def split_text(text, k):
        # 移除换行符并获取总长度
        text = text.replace('\n', '')
        total_length = len(text)
        
        # 计算每份的基本长度
        base_length = total_length // k
        
        # 创建k个基本分段
        result = []
        for i in range(k):
            start = i * base_length
            end = start + base_length if i < k-1 else total_length
            result.append(text[start:end])
        
        # 创建中间部分（每两份之间的重叠部分）
        merged_result = []
        for i in range(k-1):
            current = result[i]
            next_part = result[i+1]
            # 取当前部分的后半部分和下一部分的前半部分
            mid_section = current[len(current)//2:] + next_part[:len(next_part)//2]
            merged_result.append(mid_section)
        
        # 组合最终结果
        final_result = []
        for i in range(k-1):
            final_result.append(result[i])
            final_result.append(merged_result[i])
        final_result.append(result[-1])
        
        return final_result



def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

set_seed(42)

sample1 = "// 这里是输出的例子，保密原因删去"
sample2 = """
// 这里是输出的例子，保密原因删去
"""

# Utils
import pypandoc
def read_docx(file_path):
    output = pypandoc.convert_file(file_path, 'plain')
    return output

directory = "./data"
files = {}
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    if file_path.endswith("docx") or file_path.endswith("doc"):
        content = read_docx(file_path)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    print(f"{filename}: {len(content)}")
    files[filename] = content


engine = Model(model_path="/home/u2023202305/models/Qwen/Qwen2.5-72B-Instruct", gpu_memory_utilization=0.95)
prompt = "请你从社会学研究的角度，总结下面以<article></article>包裹的文本，要求：（1）保留关键内容；（2）使用有真实意义的一级二级标题，标题应该为一个八个字以内的短语，请尽量使用专有名词，如果不行再使用偏正短语或动补短语。\n\n<article>%%Question%%</article>\n\n"
engine.evaluation(
    inputs = files,
    NUM=13, NUM_CLUSTER=15, temperature=0.9, max_tokens=8192,
    pattern=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt.replace("%%Question%%", sample1)},
        {"role": "assistant", "content": sample2},
        {"role": "user", "content": prompt}
    ],
)

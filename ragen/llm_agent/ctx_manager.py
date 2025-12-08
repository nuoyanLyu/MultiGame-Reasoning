"""
This is the context manager for the LLM agent.
author: Kangrui Wang, Zihan Wang
date: 2025-03-30
"""
from itertools import zip_longest

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import re
import swanlab
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from transformers import AutoTokenizer
import hydra
from ragen.utils import register_resolvers
from ragen.env import REGISTERED_ENV_CONFIGS
from tensordict import TensorDict

from dataclasses import asdict
register_resolvers()

def get_special_tokens(tokenizer: AutoTokenizer):
    # if "qwen" in tokenizer.name_or_path.lower():
    # 只训练QWEN系列模型，取消对其他模型接口的支持
    # TODO：后续如果换成其他模型可能需要对应适配
    special_token = tokenizer.encode("<|im_start|>")[0]
    reward_token = tokenizer.encode("<|im_end|>")[0]
    # elif "llama-3" in tokenizer.name_or_path.lower():
    #     special_token = 128006
    #     reward_token = 128009
    # else:
    #     raise ValueError(f"Unsupported model: {tokenizer.name_or_path}")
    return special_token, reward_token

def get_masks_and_scores(input_ids: torch.Tensor, tokenizer: AutoTokenizer, all_scores: List[List[float]] = None, use_turn_scores: bool = False, enable_response_mask: bool = False):
    """
    input_ids: shape (bsz, seq_len)
    Get loss mask that only learns between <|im_start|>assistant and <|im_end|>. Currently only supports qwen.
    NOTE: important! This assumes that the input_ids starts with system and then user & assistant in alternative ways
    """
    # input_ids：输入文本的token序列
    # special_token: |im_start|, reward_token: |im_end|
    special_token, reward_token = get_special_tokens(tokenizer)
    # 返回token是否是|im_start|，是为1否则为0
    turn_starts = torch.where(input_ids == special_token, 1, 0)
    # 使用cumsum，第一个im_start开始就是1111，第二个im_start开始就是2222，以此类推
    turn_indicators = torch.cumsum(turn_starts, dim=-1)
    if enable_response_mask:  # 应该是设置为了True提高训练稳定性
        # 确定一下消息结构：依次为 system, user, assistant, user, assistant这样的结构
        loss_mask = (turn_indicators % 2 == 1) & (turn_indicators > 1) # only learns all assistant turns
        # 所以loss_mask是所有的assistant输出的信息
        # print(f"loss_mask: {loss_mask}")
    else:
        loss_mask = (turn_indicators > 1) # learns everything after system prompt
    # 同loss_mask，是assistant输出信息
    response_mask = (turn_indicators % 2 == 1) & (turn_indicators > 1)
    # print(f'response_mask: {response_mask}')
    score_tensor = torch.zeros_like(input_ids, dtype=torch.float32)
    if use_turn_scores: # 应该是设置为了False NOTE：后续如果加入step_score可能多处需要修改
        for idx, scores in enumerate(zip_longest(*all_scores, fillvalue=0)):
            scores = torch.tensor(scores, dtype=torch.float32)
            turn_indicator = idx * 2 + 3 # 0: pad. 1: system. 2+2n: user. 3+2n: assistant
            reward_position = (input_ids == reward_token) & (turn_indicators == turn_indicator)
            # Set the last token of the rows where all positions are False to True
            reward_position[~reward_position.any(dim=-1), -1] = True
            score_tensor[reward_position] = scores
        if "qwen" in tokenizer.name_or_path.lower():
            # for Qwen, there is a "\n" between special token and reward token, so we shift this to make sure reward is assigned to the last token of a turn
            score_tensor = score_tensor.roll(shifts=1, dims=-1)
    else:
        # print(f"all_scores: {all_scores}")
        scores = [sum(i) for i in all_scores]  # 所有数据的加一起
        score_tensor[:, -1] = torch.tensor(scores, dtype=torch.float32)
    score_tensor = score_tensor[:, 1:] # remove the first token
    loss_mask = loss_mask[:, :-1] # remove the last token
    response_mask = response_mask[:, :-1] # remove the last token

    return score_tensor, loss_mask, response_mask


class ContextManager:
    """
    Manages the context for LLM interactions with environments.
    Translates between environment outputs and LLM inputs, and vice versa.
    """

    def __init__(self, 
                 config,
                 tokenizer,
                 processor = None,
                 mode: str = "train",
                 ):
        """
        Initialize the ContextManager.
        Processor is used to process the image data.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.action_sep = self.config.agent_proxy.action_sep
        self.special_token_list = ["<think>", "</think>", "<answer>", "</answer>", "<|im_start|>", "<|im_end|>"]

        self.es_cfg = self.config.es_manager[mode]
        self.env_nums = {
                env_tag: n_group * self.es_cfg.group_size
                for n_group, env_tag in zip(self.es_cfg.env_configs.n_groups, self.es_cfg.env_configs.tags)
        }
        self._init_prefix_lookup()
    
    def _check_env_installed(self, env_type: str):
        if env_type not in REGISTERED_ENV_CONFIGS:
            raise ValueError(f"Environment {env_type} is not installed. Please install it using the scripts/setup_{env_type}.sh script.")

    def _init_prefix_lookup(self):
        prefix_lookup = {}
        prefixes = {}
        env_config_lookup = {}
        env_config = {}
        for env_tag, env_config in self.config.custom_envs.items():
            if env_tag not in self.es_cfg.env_configs.tags:
                continue

            self._check_env_installed(env_config.env_type)
            env_config_new = asdict(REGISTERED_ENV_CONFIGS[env_config.env_type]())
            for k,v in env_config.items():
                env_config_new[k] = v
            # 如果有env_multi_instruction，提取对应的表格并随机抽取一个作为env_instruct
            # 但是这里可能也要控制随机性——需要保证相同种子能够生成相同的prompt信息
            env_instruction = env_config_new.get("env_instruction", "")
            if env_config_new.get("grid_vocab", False):
                grid_vocab_str = "\nThe meaning of each symbol in the state is:\n" + ", ".join([f"{k}: {v}" for k, v in env_config_new["grid_vocab"].items()])
                env_instruction += grid_vocab_str
            if env_config_new.get("action_lookup", False):
                action_lookup_str = "\nYour available actions are:\n" + ", ".join([f"{v}" for k, v in env_config_new["action_lookup"].items()])
                action_lookup_str += f"\nYou can make up to {env_config_new['max_actions_per_traj']} actions, separated by the action separator \" " + self.action_sep + " \"\n"
                env_instruction += action_lookup_str
            prefixes[env_tag] = env_instruction
            env_config_lookup[env_tag] = {'max_tokens': env_config.get("max_tokens", self.config.actor_rollout_ref.rollout.response_length)}

        tags = self.es_cfg.env_configs.tags
        n_groups = self.es_cfg.env_configs.n_groups
        group_size = self.es_cfg.group_size

        cur_group = 0
        for env_tag, n_group in zip(tags, n_groups):
            env_instruction = prefixes[env_tag]
            start_idx = cur_group * group_size
            end_idx = (cur_group + n_group) * group_size
            for i in range(start_idx, end_idx):
                prefix_lookup[i] = env_instruction
                env_config_lookup[i] = env_config_lookup[env_tag]
            cur_group += n_group
            
        self.prefix_lookup = prefix_lookup
        self.env_config_lookup = env_config_lookup

    def _parse_response_ori(self, response: str) -> List:
        pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>' if self.config.agent_proxy.enable_think else r'<answer>(.*?)</answer>'
        match = re.search(pattern, response, re.DOTALL)
        if not match:
            # think_content, action_content, actions = "", "", [] # do not remove this kind of invalid string
            llm_response, actions = response, []
        else:
            if self.config.agent_proxy.enable_think:
                think_content, action_content = match.group(1), match.group(2)
            else:
                think_content, action_content = "", match.group(1)

                
            for special_token in self.special_token_list:
                action_content = action_content.replace(special_token, "").strip()
                think_content = think_content.replace(special_token, "").strip()
            
            actions = [action.strip() for action in action_content.split(self.action_sep) if action.strip()]
            max_actions = self.config.agent_proxy.max_actions_per_turn

            if len(actions) > max_actions:
                actions = actions[:max_actions] #Only the first MAX_ACTIONS actions are kept in the rollout.
                action_content = (" " + self.action_sep + " ").join(actions)

            llm_response = f"<think>{think_content}</think><answer>{action_content}</answer>" if self.config.agent_proxy.enable_think else f"<answer>{action_content}</answer>"
        return llm_response, actions

    def _parse_response(self, response: str) -> List:
        pattern = r'.*<think>(.*?)</think>\s*<answer>(.*?)</answer>$' if self.config.agent_proxy.enable_think else r'.*<answer>(.*?)</answer>$'
        match = re.match(pattern, response, re.DOTALL)
        # pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>' if self.config.agent_proxy.enable_think else r'<answer>(.*?)</answer>'
        # match = re.search(pattern, response, re.DOTALL)
        if not match:
            # think_content, action_content, actions = "", "", [] # do not remove this kind of invalid string
            llm_response, actions = response, []
        else:
            if self.config.agent_proxy.enable_think:
                think_content, action_content = match.group(1), match.group(2)
            else:
                think_content, action_content = "", match.group(1)
                
            for special_token in self.special_token_list:
                action_content = action_content.replace(special_token, "").strip()
                think_content = think_content.replace(special_token, "").strip()
            
            actions = [action.strip() for action in action_content.split(self.action_sep) if action.strip()]
            max_actions = self.config.agent_proxy.max_actions_per_turn

            if len(actions) > max_actions:
                actions = actions[:max_actions] #Only the first MAX_ACTIONS actions are kept in the rollout.
                action_content = (" " + self.action_sep + " ").join(actions)
            # 保留之前的打包方案，仅保留符合格式的内容，和原始处理方法相同
            llm_response = f"<think>{think_content}</think><answer>{action_content}</answer>" if self.config.agent_proxy.enable_think else f"<answer>{action_content}</answer>"
            # llm_response = response
        return llm_response, actions
        
    def _normalize_score_tensor(self, score_tensor: torch.Tensor, env_outputs: List[Dict]) -> torch.Tensor:
        """
        Normalize the score tensor to be between 0 and 1.
        NOTE: only support score at the last token for now
        """
        assert self.config.agent_proxy.use_turn_scores == False, "Reward normalization is not supported for use_turn_scores == True"
        
        rn_cfg = self.config.agent_proxy.reward_normalization
        grouping, method = rn_cfg.grouping, rn_cfg.method
        if grouping == "state":
            group_tags = [env_output["group_id"] for env_output in env_outputs]
        elif grouping == "inductive":
            group_tags = [env_output["tag"] for env_output in env_outputs]
        elif grouping == "batch":
            group_tags = [1] * len(env_outputs)
        else:
            raise ValueError(f"Invalid grouping: {grouping}")


        if method == "mean_std":
            norm_func = lambda x: (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6) if x.std(dim=-1, keepdim=True).abs().max() > 1e-6 else torch.zeros_like(x) # stable to bf16 than x.std()
        elif method == "mean":
            norm_func = lambda x: (x - x.mean(dim=-1, keepdim=True))
        elif method == "asym_clip":
            norm_func = lambda x: ((x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6) if x.std(dim=-1, keepdim=True).abs().max() > 1e-6 else torch.zeros_like(x)).clamp(min=-1, max=3)
        elif method == "identity":
            norm_func = lambda x: x
        else:
            raise ValueError(f"Invalid normalization method: {method}")

        # apply groupwise normalization
        group2index = {}
        for i, env_tag in enumerate(group_tags):
            if env_tag not in group2index:
                group2index[env_tag] = []
            group2index[env_tag].append(i)
        group2index = {k: torch.tensor(v) for k, v in group2index.items()}

        
        # apply penalty pre-normalization 在这里会把之前的penalty项考虑进去
        acc_scores = score_tensor[:, -1]
        normalized_acc_scores = acc_scores.clone()
        penalty = torch.tensor([env_output.get("penalty", 0) for env_output in env_outputs], dtype=torch.float32)
        normalized_acc_scores = normalized_acc_scores + penalty

        if len(group2index) < acc_scores.shape[0]: # the group size > 1
            for group, index in group2index.items():
                normalized_acc_scores[index] = norm_func(normalized_acc_scores[index])
        # print("normalized_acc_scores: ", normalized_acc_scores)
        score_tensor[:, -1] = normalized_acc_scores

        return score_tensor
    
    def get_lm_inputs_ori(self, env_outputs: List[Dict], prepare_for_update: bool) -> DataProto:
        """
        env_outputs - please see below example
        [
            {"env_id": 1, "history": [{"state": "###\n#x_#", "llm_response": "Response 1", "reward": 0.5}, {"state": "###\n#x_#"}]},
            {"env_id": 2, "history": [{"state": "###\n#x_#"}]},
            ...
        ]
        prefix_lookup - from env_id to initial prompt
        """
        llm_input_texts = []
        messages_list = [] # for api calling
        for env_output in env_outputs:
            if 'state' in env_output['history'][-1] and prepare_for_update:
                env_output['history'] = env_output['history'][:-1] 
                # when prepare for update, we do not add the state from the n+1 turn to the trajectory
                # 合理的，最后一次的state也没有动作、没有意义
            
            max_k = getattr(self.config.agent_proxy, "max_context_window", None)
            if max_k is not None and isinstance(max_k, int) and max_k > 0:
                env_output['history'] = env_output['history'][-max_k:]
            
            messages = [
                {"role": "system", "content": f"You're a helpful assistant. "}, 
                {"role": "user", "content": self.prefix_lookup[env_output["env_id"]]}
            ]
            # 有个问题——训练的时候模型是否能看到前几轮的历史信息？应不应该看到历史信息？
            for idx, content in enumerate(env_output["history"]):
                # 陆续打包history里面包含的历史信息
                # messages[-1]['content]是最后一个user对应的content信息
                messages[-1]["content"] += f"\nTurn {idx + 1}:\n"
                if "state" in content:
                    FORMAT_PROMPT = "<think> [Your thoughts] </think> <answer> [your answer] </answer>" if self.config.agent_proxy.enable_think else "<answer> [your answer] </answer>"
                    LENGTH_PROMPT = f"Max response length: {self.env_config_lookup[env_output['env_id']]['max_tokens']} words (tokens)."
                    messages[-1]["content"] += f"State:\n{content['state']}\nYou have {content['actions_left']} actions left. Let\'s think step by step and always output: {FORMAT_PROMPT} with no extra text. Strictly follow this format. {LENGTH_PROMPT}\n"
                # prepare for update可能才有？里面包含了历史的LLM输出
                if "llm_response" in content:
                    # 打包历史的llm回复信息——这里用的不是原始消息——这样对吗？
                    # 如果直接不符合格式是不是就会结束、不继续输出，如果按照这个逻辑直接拼进来llm_response也没问题
                    messages.append({"role": "assistant", "content": content["llm_response"]})
                if "reward" in content and not (prepare_for_update and idx == len(env_output["history"]) - 1):
                    # when prepare for update, we do not add the reward from the n+1 turn to the trajectory
                    messages.append({"role": "user", "content": f"Reward:\n{content['reward']}\n"})
                    
            # NOTE: this assertion is important for loss mask computation  
            # 但是实际的计算中实际上并没有通过history包含一些信息，可能只有一条system以及user信息 
            # 这个对吗？check一下别的环境的history信息如何设置，多进行几轮多check一下
            # 还有个问题，这个形式是不是就是要求模型不能中间有reward信息？
            assert all(msg["role"] == "assistant" for msg in messages[2::2])
            # 所以是通过assistant计算的loss并不是通过<think>之类的token——也合理
            text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=(not prepare_for_update), tokenize=False)
            # NOTE：这个应该要进行修改，这种设定并不合理，整个chat_template都应该修改
            if not prepare_for_update:
                if self.config.agent_proxy.enable_think:
                    # NOTE：删掉，一般的指令遵循不会有这个<think>的token
                    text += "<think>" # force the LLM to think before answering
                else:
                    text += "<answer>" # force the LLM to answer
            llm_input_texts.append(text)
            messages_list.append(messages)
        # 打一下日志
        # input_swan = swanlab.Text(llm_input_texts[0])
        # swanlab.log({"input_text": input_swan})
        # print(llm_input_texts[0])
        inputs = self.tokenizer(llm_input_texts, return_tensors="pt", padding=True, padding_side="left", truncation=False) # do not truncate here. Process later at TODO
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        position_ids = attention_mask.cumsum(dim=-1)
        if prepare_for_update:
            scores = [[i.get('reward', 0.0) for i in env_output['history']] for env_output in env_outputs]
            score_tensor, loss_mask, response_mask = get_masks_and_scores(input_ids, self.tokenizer, scores, use_turn_scores=self.config.agent_proxy.use_turn_scores, enable_response_mask=self.config.enable_response_mask)

            normalized_score_tensor = score_tensor
            if not self.config.agent_proxy.use_turn_scores:
                normalized_score_tensor = self._normalize_score_tensor(score_tensor, env_outputs)
            response_length = response_mask.sum(dim=-1).float().mean().item()

        llm_inputs = DataProto()
        llm_inputs.batch = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": input_ids[:, 1:], # remove the first token
        }, batch_size=input_ids.shape[0])
        # 更像是输入的里面有多少是真正的参与loss计算的response信息

        if prepare_for_update:
            llm_inputs.batch["loss_mask"] = loss_mask # remove the first token
            llm_inputs.batch["rm_scores"] = normalized_score_tensor # remove the first token
            llm_inputs.batch["original_rm_scores"] = score_tensor # remove the first token
        
        llm_inputs.non_tensor_batch = {
            "env_ids": np.array([env_output["env_id"] for env_output in env_outputs], dtype=object),
            "group_ids": np.array([env_output["group_id"] for env_output in env_outputs], dtype=object),
            "messages_list": np.array(messages_list, dtype=object),
        }

        if prepare_for_update:
            metrics = {}
            for env_output in env_outputs:
                for key, value in env_output["metrics"].items():
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
            mean_metrics = {
                key: np.sum(value) / self.env_nums[key.split("/")[0]]
                for key, value in metrics.items()
            }
            for key, values in metrics.items():
                if not isinstance(values, list):
                    continue
                prefix, suffix = key.split("/", 1)
                non_zero_values = [v for v in values if v != 0]
                if non_zero_values:  # Avoid division by zero
                    non_zero_key = f"{prefix}/non-zero/{suffix}"
                    mean_metrics[non_zero_key] = np.mean(non_zero_values)
            metrics = mean_metrics
            metrics["response_length"] = response_length
            llm_inputs.meta_info = {"metrics": metrics}
        return llm_inputs

    def get_lm_inputs(self, env_outputs: List[Dict], prepare_for_update: bool) -> DataProto:
        """
        env_outputs - please see below example
        [
            {"env_id": 1, "history": [{"state": "###\n#x_#", "llm_response": "Response 1", "reward": 0.5}, {"state": "###\n#x_#"}]},
            {"env_id": 2, "history": [{"state": "###\n#x_#"}]},
            ...
        ]
        prefix_lookup - from env_id to initial prompt
        """
        llm_input_texts = []
        messages_list = [] # for api calling
        for env_output in env_outputs:
            if 'state' in env_output['history'][-1] and prepare_for_update:
                env_output['history'] = env_output['history'][:-1] 
                # when prepare for update, we do not add the state from the n+1 turn to the trajectory
                # 合理的，最后一次的state也没有动作、没有意义
            
            max_k = getattr(self.config.agent_proxy, "max_context_window", None)
            if max_k is not None and isinstance(max_k, int) and max_k > 0:
                env_output['history'] = env_output['history'][-max_k:]
            
            messages = [
                {"role": "system", "content": f"You're a helpful assistant. "}, 
                {"role": "user", "content": self.prefix_lookup[env_output["env_id"]]}
            ]
            # 有个问题——训练的时候模型是否能看到前几轮的历史信息？应不应该看到历史信息？
            for idx, content in enumerate(env_output["history"]):
                # 陆续打包history里面包含的历史信息
                # messages[-1]['content]是最后一个user对应的content信息
                messages[-1]["content"] += f"\nTurn {idx + 1}:\n"
                if "state" in content:
                    FORMAT_PROMPT = "<think> [Your thoughts] </think> <answer> [your answer] </answer>" if self.config.agent_proxy.enable_think else "<answer> [your answer] </answer>"
                    LENGTH_PROMPT = f"Max response length: {self.env_config_lookup[env_output['env_id']]['max_tokens']} words (tokens)."
                    messages[-1]["content"] += f"State:\n{content['state']}\nYou have {content['actions_left']} actions left. Let\'s think step by step and always output: {FORMAT_PROMPT} with no extra text. Strictly follow this format. {LENGTH_PROMPT}\n"
                # prepare for update可能才有？里面包含了历史的LLM输出
                if "llm_response" in content:
                    # 打包历史的llm回复信息——打包的是处理后的结果
                    messages.append({"role": "assistant", "content": content["llm_response"]})
                if "reward" in content and not (prepare_for_update and idx == len(env_output["history"]) - 1):
                    # when prepare for update, we do not add the reward from the n+1 turn to the trajectory
                    messages.append({"role": "user", "content": f"Reward:\n{content['reward']}\n"})
                    
            # NOTE: this assertion is important for loss mask computation  
            # 但是实际的计算中实际上并没有通过history包含一些信息，可能只有一条system以及user信息 
            # 这个对吗？check一下别的环境的history信息如何设置，多进行几轮多check一下
            # 还有个问题，这个形式是不是就是要求模型不能中间有reward信息？
            assert all(msg["role"] == "assistant" for msg in messages[2::2])
            # 所以是通过assistant计算的loss并不是通过<think>之类的token——也合理
            text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=(not prepare_for_update), tokenize=False)
            # NOTE：这个应该要进行修改，这种设定并不合理，不需要在最后强制加上<think>的token
            # if not prepare_for_update:
            #     if self.config.agent_proxy.enable_think:
            #         # NOTE：删掉，一般的指令遵循不会有这个<think>的token
            #         text += "<think>" # force the LLM to think before answering
            #     else:
            #         text += "<answer>" # force the LLM to answer
            llm_input_texts.append(text)
            messages_list.append(messages)
        inputs = self.tokenizer(llm_input_texts, return_tensors="pt", padding=True, padding_side="left", truncation=False) # do not truncate here. Process later at TODO
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
        # print(position_ids)  这个position+ids我之前修改过吗……确认一下，总感觉不对
        if prepare_for_update:
            # 看一下打包好的input结果
            scores = [[i.get('reward', 0.0) for i in env_output['history']] for env_output in env_outputs]
            score_tensor, loss_mask, response_mask = get_masks_and_scores(input_ids, self.tokenizer, scores, use_turn_scores=self.config.agent_proxy.use_turn_scores, enable_response_mask=self.config.enable_response_mask)

            normalized_score_tensor = score_tensor
            if not self.config.agent_proxy.use_turn_scores:
                normalized_score_tensor = self._normalize_score_tensor(score_tensor, env_outputs)
            response_length = response_mask.sum(dim=-1).float().mean().item()

        llm_inputs = DataProto()
        llm_inputs.batch = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": input_ids[:, 1:], # remove the first token
        }, batch_size=input_ids.shape[0])

        if prepare_for_update:
            # 这些remove token是为了predict next token的loss计算，和<think>的token五官
            llm_inputs.batch["loss_mask"] = loss_mask # remove the first token
            llm_inputs.batch["rm_scores"] = normalized_score_tensor # remove the first token
            llm_inputs.batch["original_rm_scores"] = score_tensor # remove the first token
        
        llm_inputs.non_tensor_batch = {
            "env_ids": np.array([env_output["env_id"] for env_output in env_outputs], dtype=object),
            "group_ids": np.array([env_output["group_id"] for env_output in env_outputs], dtype=object),
            "messages_list": np.array(messages_list, dtype=object),
        }

        if prepare_for_update:
            metrics = {}
            for env_output in env_outputs:
                for key, value in env_output["metrics"].items():
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
            print(metrics)
            # 这个地方对相同env的metric都计算了平均值——比如每个env有16个group，每个group有16个，
            # 返回的是同一个group的metric，而后再对这16个求平均值
            mean_metrics = {
                key: np.sum(value) / self.env_nums[key.split("/")[0]]
                for key, value in metrics.items()
            }
            for key, values in metrics.items():
                if not isinstance(values, list):
                    continue
                prefix, suffix = key.split("/", 1)
                non_zero_values = [v for v in values if v != 0]
                if non_zero_values:  # Avoid division by zero
                    non_zero_key = f"{prefix}/non-zero/{suffix}"
                    mean_metrics[non_zero_key] = np.mean(non_zero_values)
            metrics = mean_metrics
            metrics["response_length"] = response_length
            llm_inputs.meta_info = {"metrics": metrics}
        return llm_inputs

    def get_env_inputs_ori(self, lm_outputs: DataProto) -> List[Dict]:
        if lm_outputs.batch is not None and 'responses' in lm_outputs.batch.keys():
            responses = self.tokenizer.batch_decode(
                lm_outputs.batch['responses'], 
                skip_special_tokens=True
            )
        else: # dataproto has textual responses
            responses = lm_outputs.non_tensor_batch['response_texts']
        # NOTE：这个地方把<think>这个token加回去了，后续要删除
        responses = ["<think>" + response if self.config.agent_proxy.enable_think else "<answer>" + response for response in responses] # The LLM generation does not include <think> tags. Add them back here.
        env_ids = lm_outputs.non_tensor_batch['env_ids']
        env_inputs = []
        for env_id, response in zip(env_ids, responses):
            llm_response, actions = self._parse_response(response)
            env_inputs.append({
                "env_id": env_id,
                "llm_raw_response": response,
                "llm_response": llm_response, 
                # 改过后llm_response就是llm_raw_response，确定除了log之外这个llm_response都没有用到
                "actions": actions,
            })


    def get_env_inputs(self, lm_outputs: DataProto) -> List[Dict]:
        if lm_outputs.batch is not None and 'responses' in lm_outputs.batch.keys():
            responses = self.tokenizer.batch_decode(
                lm_outputs.batch['responses'], 
                skip_special_tokens=True
            )
        else: # dataproto has textual responses
            responses = lm_outputs.non_tensor_batch['response_texts']
        # NOTE：这个地方把<think>这个token加回去了，删除
        # responses = ["<think>" + response if self.config.agent_proxy.enable_think else "<answer>" + response for response in responses] # The LLM generation does not include <think> tags. Add them back here.
        env_ids = lm_outputs.non_tensor_batch['env_ids']
        env_inputs = []
        for env_id, response in zip(env_ids, responses):
            llm_response, actions = self._parse_response(response)
            env_inputs.append({
                "env_id": env_id,
                "llm_raw_response": response,
                "llm_response": llm_response,
                "actions": actions,
            })

        return env_inputs

    def formulate_rollouts(self, env_outputs: List[Dict]) -> DataProto:
        llm_inputs = self.get_lm_inputs(env_outputs, prepare_for_update=True)
        return llm_inputs

    

@hydra.main(version_base = None, config_path = "../../config", config_name = "base")
def main(config):
    import json
    # batch_list = [
    #     {
    #         "env_ids": 0,
    #         "chat_response": "<think><think></answer> 123. </think><answer> <answer> say | hi </answer></answer>",
    #     },
    #     {
    #         "env_ids": 1,
    #         "chat_response": "<think> 456. </think><answer> 789 </answer><think> 10123 </think><answer> 11111 </answer>",
    #     }
    # ]
    # ctx_manager.action_sep_lookup = {
    #     0: "|",
    #     1: ";"
    # }
    # for item in batch_list:
    #     item["responses"] = tokenizer.encode(item["chat_response"], return_tensors="pt",max_length=512, truncation=True,padding="max_length")[0]
    # batch_dict = collate_fn(batch_list)
    # batch = DataProto.from_single_dict(batch_dict)
    # env_inputs = ctx_manager.get_env_inputs(batch)
    # print(env_inputs)
    


    env_outputs = [
        {
            "env_id": 1,
            "history": [
                {"state": "###\n#x_#<image>", "llm_response": "Response 1", "reward": 0.5, "actions_left": 2},
                {"state": "###\n#x_#<image>", "llm_response": "Response 2", "reward": 0.8, "actions_left": 1},
                {"state": "###\n#x_#<image>", "actions_left": 0}
            ],
            "group_id": 0,
            "metrics": {}
        },
        {
            "env_id": 2,
            "history": [
                {"state": "Test", "llm_response": "Response 3", "reward": 0.3, "actions_left": 1},
                {"state": "Test", "actions_left": 0}
            ],
            "group_id": 1,
            "metrics": {}
        }
    ]
    
    prefix_lookup = {1: "Initial prompt", 2: "Initial prompt 2"}
    tokenizer = AutoTokenizer.from_pretrained("/data1/lvnuoyan/llm_model/Qwen2.5-1.5B-Instruct")
    ctx_manager = ContextManager(config=config, tokenizer=tokenizer)
    # print("ctx_manager prefix", ctx_manager.prefix_lookup)
    env_prompt = ctx_manager.get_lm_inputs(env_outputs, prepare_for_update=False)
    # print(env_prompt[0].batch['responses'])
    formulate_rollouts_rst= ctx_manager.formulate_rollouts(env_outputs)
    # print(formulate_rollouts_rst[0])

if __name__ == "__main__":
    main()
    

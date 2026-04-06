import time
import os
from .utils import deal_answer, deal_judge, deal_post, str2paras, deal_judge_new, has_answer, has_answer_by_llm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from tqdm import tqdm
from llms import llm_client as llm_client
all_choices=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


os.environ['CUDA_VISIBLE_DEVICES'] = "0"


class Generater:
    def __init__(self, args):
        self.args = args
        model_id = "microsoft/Phi-3-mini-4k-instruct"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        self.batch_size = args.batch_size
        self.outputs = []
        self.args.model_name = "phi-3-mini-4k-instruct" 
        self.eos_id_dict = {
            'llama2-7b-chat': self.tokenizer.eos_token_id,
            'llama3-8b-instruct': self.tokenizer.convert_tokens_to_ids(['<|eot_id|>'])[0],
            'qwen2-7b-instruct': self.tokenizer.eos_token_id,
            'llama2-13b-chat': self.tokenizer.eos_token_id,
            'phi-3-mini-4k-instruct': self.tokenizer.eos_token_id,
        }
        print('load generater finish.')

    def load_data(self, data):
        self.data = data
        self.dataloader = DataLoader(self.data, shuffle=False, batch_size=self.batch_size)
        if self.args.task == 'mmlu' or self.args.task == 'tq':
            self.choice_cnt = self.data.choice_cnt

    def get_res(self):
        self.outputs = []
        device = torch.device('cuda')
        self.device = device
        self.model.to(device)
        for batch in tqdm(self.dataloader):
            batch = self.tokenizer(batch, return_tensors='pt', padding=True).to(device)
            input_ids, attn_mask = batch['input_ids'], batch['attention_mask']
            outs = self.model.generate(input_ids, attention_mask=attn_mask, max_new_tokens=self.args.max_new_tokens,
                                       output_attentions=self.args.attn_weights, return_dict_in_generate=True, output_scores=True, output_hidden_states=self.args.hidden_states,
                                       pad_token_id=0, top_p=1.0, temperature=1, do_sample=False)
            if self.args.task == 'mmlu' or self.args.task == 'tq':
                self.process_res_multi_choice(outs, input_ids) 
            else:
                self.process_res(outs, input_ids)
        print(f'len of outputs: {len(self.outputs)}')
        return self.calculate_res()
    
    def process_res(self, outs, inputs):
        
        scores = outs['scores']    
        seqs = outs['sequences']    
        input_len = inputs.shape[-1]
        bt_size = inputs.shape[0]
        new_ids = seqs[:, input_len:]   
        end_idx = self.get_generation_end(new_ids)
        print(f'end_idx: {end_idx}')
        top_indices, top_scores, ans_scores, ans_entropy = self.get_generated_tokens_probs_entropy(scores, new_ids, bt_size)

        if self.args.hidden_states:
            hidden_modes = self.args.hidden_idx_mode.split(',')
            all_modes_hidden_state = [{} for _ in range(bt_size)]
            for mode in hidden_modes:
                if mode == 'ans':   
                    raise ValueError('Do not support hidden_mode=ans for free-form qa')
                if mode == 'every':     
                    probs_for_generated_tokens, tokens_for_each_layer = self.get_token_and_prob_for_each_pos(outs, bt_size, end_idx) #(bt_size, layers, ans_len)
                else:
                    if mode == 'conf':
                        pos_idx = self.get_confidence_idx(outs, inputs, end_idx)
                    else:
                        pos_idx = self.get_need_idx_for_generation(top_scores, end_idx, mode)
                    hidden_states = self.get_hidden_states_for_given_pos(outs, bt_size, pos_idx, mode)
                    for bt in range(bt_size):
                        all_modes_hidden_state[bt][mode] = hidden_states[bt]
                

        for bt in range(bt_size):
            # print(f'ans: {self.tokenizer.decode(new_ids[bt][:end_idx[bt]])}')
            temp_res = ({
                'Res': self.tokenizer.decode(new_ids[bt][:end_idx[bt]]).strip(),
                'Log_p':{
                    'tokens': new_ids[bt][:end_idx[bt]].tolist(),
                    'token_probs': ans_scores[bt][:end_idx[bt]].tolist(),
                    'token_entropy': ans_entropy[bt][:end_idx[bt]].tolist()
                }
            })
            if self.args.hidden_states:
                if self.args.hidden_idx_mode == 'every':
                    temp_res['probs_for_generated_tokens'] = probs_for_generated_tokens[bt]
                    temp_res['tokens_for_each_layer'] = tokens_for_each_layer[bt]
                else:
                    temp_res['hidden_states'] = all_modes_hidden_state[bt]

            self.outputs.append(temp_res)

    def process_res_multi_choice(self, outs, inputs):

        choices = all_choices[:self.choice_cnt] + all_choices[:self.choice_cnt]
        if self.args.model_name in ['llama3-8b-instruct', 'qwen2-7b-instruct']:
            choices = all_choices[:self.choice_cnt] + all_choices[:self.choice_cnt] + all_choices[:self.choice_cnt]
        input_len = inputs.shape[-1]
        seqs = outs['sequences'] 
        scores = outs['scores'] # tuple of tensor (generated_len) -> (batch_size, vocab_size)
        new_ids = seqs[:, input_len:] # batch_size, new_seq_len
        end_idx = self.get_generation_end(new_ids)
        print(f'end idx: {end_idx}')
        ans_token_idx, choices_idx = self.get_choice_idx(outs, inputs, end_idx)
        print(f'answer idx: {ans_token_idx}')
        need_scores = []
        bt_size = inputs.shape[0]
        for bt in range(bt_size):
            need_scores.append(scores[ans_token_idx[bt]][bt]) # vocab_size
        need_scores = torch.stack(need_scores)
        probs = nn.Softmax(dim=-1)(need_scores) 
        next_token_probs = probs[:, choices_idx] # batch_size, 8
        entropy = torch.sum(-(probs * torch.log2(probs)), dim=-1) # batch_size, 8
        max_scores, max_indices = torch.max(next_token_probs, dim=-1) 
        _, top_scores, _, _ = self.get_generated_tokens_probs_entropy(scores, new_ids, bt_size)

        if self.args.attn_weights: 
            attentions = self.get_attn_multi_choice(outs, bt_size, ans_token_idx)

        if self.args.hidden_states:
            hidden_modes = self.args.hidden_idx_mode.split(',')
            all_modes_hidden_state = [{} for _ in range(bt_size)]
            for mode in hidden_modes:
                if mode == 'every': 
                    raise ValueError('Do not need to specify hidden_idx_mode=every for multi-choice qa')
                elif mode == 'ans':
                    hidden_states = self.get_hidden_states_for_given_pos(outs, bt_size, ans_token_idx, mode)
                else:
                    if mode == 'conf':
                        pos_idx = self.get_confidence_idx(outs, inputs, end_idx)
                    else:
                        pos_idx = self.get_need_idx_for_generation(top_scores, end_idx, mode)
                    hidden_states = self.get_hidden_states_for_given_pos(outs, bt_size, pos_idx, mode)
                for bt in range(bt_size):
                    all_modes_hidden_state[bt][mode] = hidden_states[bt]
        
        for bt in range(bt_size):
            temp_res = {
                'Res': choices[max_indices[bt]],
                'Full_res': self.tokenizer.decode(new_ids[bt][:end_idx[bt]]).strip(),
                'Log_p':{
                    'token probs': next_token_probs[bt].tolist(),# choices prob
                    'token_entropy': float(entropy[bt]), # real entropy
                },
                'end_idx': end_idx[bt]
            }
            if self.args.hidden_states:
                temp_res['hidden_states'] = all_modes_hidden_state[bt]
            if self.args.output_states:
                temp_res['output_states'] = probs[bt]
            if self.args.attn_weights:
                temp_res['attn_weights'] = attentions[bt]
            self.outputs.append(temp_res)

    def calculate_res(self):
        """Save the output results."""
        all_data = self.data.data 
        res = []
        begin = 0
        acc = 0
        error_label = []
        print(f'len of all data: {len(all_data)}')
        for idx in range(len(all_data)):
            if idx not in self.data.idxs:  
                res.append(all_data[idx])
            else:
                res_sample = {}
                if 'qa' in self.args.type:
                    res_sample['qa_prompt'] = self.data[begin]
                    res_sample['Res'] = self.outputs[begin]['Res']
                    res_sample['Log_p'] = self.outputs[begin]['Log_p']
                    if self.args.task == 'mmlu' or self.args.task == 'tq':
                        res_sample['question'] = self.data.format_example(all_data, idx, include_answer=False)
                        res_sample['has_answer'] = res_sample['Res'] == all_data[idx][-1]
                        res_sample['reference'] = all_data[idx][-1]
                        res_sample['end_idx'] = self.outputs[begin]['end_idx']
                        res_sample['Full_res'] = self.outputs[begin]['Full_res']
                    else:
                        res_sample['question'] = all_data[idx]['question']
                        res_sample['has_answer'] = 1
                        # res_sample['has_answer'] = has_answer(all_data[idx]['reference'], res_sample['Res'])
                        # answer_label = has_answer_by_llm(res_sample['question'], all_data[idx]['reference'], res_sample['Res'], llm_client)
                        # if answer_label == -1:
                        #     print(f"Can`t check {res_sample['question']} answer label!")
                        #     error_label.append({"question": all_data[idx]['question'], "reference": all_data[idx]['reference'], "Res": res_sample['Res']})
                        #     begin += 1
                        #     continue
                        # res_sample['has_answer'] = answer_label
                        # res_sample['reference'] = all_data[idx]['reference']
                        res_sample['reference'] = ""
                    if 'prior' in self.args.type or 'post' in self.args.type: # verbalized confidence
                        res_sample['has_answer'] = deal_judge_new(res_sample['Res']) if 'mc' not in self.args.type else deal_judge_new(res_sample['Full_res'])
                    if self.args.attn_weights:
                        res_sample['attn_weights'] = self.outputs[begin]['attn_weights'].tolist()
                    if self.args.hidden_states:
                        if self.args.hidden_idx_mode == 'every':
                            res_sample['probs_for_generated_tokens'] = self.outputs[begin]['probs_for_generated_tokens']
                            res_sample['tokens_for_each_layer'] = self.outputs[begin]['tokens_for_each_layer']
                        else:
                            res_sample['hidden_states'] = self.outputs[begin]['hidden_states']
                    if self.args.output_states:
                        res_sample['output_states'] = self.outputs[begin]['output_states'].tolist()
                    acc += res_sample['has_answer']
                res.append(res_sample)
                begin += 1
        print(f'processed data count: {begin}')
        print(f'accuracy: {acc / begin}')
        return res, acc / begin, error_label
    
    def get_hidden_states_for_given_pos(self, outs, bt_size, need_idx, mode='first'):

        if self.args.need_layers == 'last':
            need_layers = [-1]
        elif self.args.need_layers == 'all':
            need_layers = range(len(outs['hidden_states'][0]))
        elif self.args.need_layers == 'mid':
            need_layers = [int(len(outs['hidden_states'][0]) / 2)]
        else:
            raise ValueError('Specify the wrong need_layers')
        # print(need_layers)
        
        res = [[] for _ in range(bt_size)]
        for bt in range(bt_size):  # Iterate over samples.
            temp_idx = need_idx[bt]  # Token index to inspect for the current sample.
            # print(f'need layers: {need_layers}')
            if type(temp_idx) != list:  # Only one token is needed.
                for layer in need_layers:  # Each layer for that token.
                    hidden_states = outs['hidden_states'][temp_idx][layer][bt][-1]  # bs, generated_len(input_len or 1), hidden_size
                    res[bt].append(hidden_states.to(torch.float16).tolist())
            else:  # Use all tokens.
                for layer in need_layers:  # Each layer for those tokens.
                    temp_res = []
                    for item in temp_idx:  # All tokens to consider.
                        temp_res.append(outs['hidden_states'][item][layer][bt][-1])
                    temp_res = torch.stack(temp_res)
                    if mode == 'avg':
                        res[bt].append(torch.mean(temp_res, dim=0).to(torch.float16).tolist())
                    elif mode == 'dim_min':  # Take the minimum across hidden dimensions.
                        res[bt].append(torch.min(temp_res, dim=0)[0].to(torch.float16).tolist())
                    elif mode == 'dim_max':
                        res[bt].append(torch.max(temp_res, dim=0)[0].to(torch.float16).tolist())
        return res

    def get_attn_multi_choice(self, outs, bt_size, need_idx):
        res = [[] for _ in range(bt_size)]
        for bt in range(bt_size):
            temp_idx = need_idx[bt]
            for layer in range(len(outs['attentions'][temp_idx])):  # All layers corresponding to the token at temp_idx.
                attentions = outs['attentions'][temp_idx][layer][bt, :, -1] # bs, head_num, seq_len(input_len)
                res[bt].append(attentions.tolist())
        return res

    def get_choice_idx(self, outs, inputs, end_idx):
        """Find the position where the choice first appears in each sample."""
        batch_size, input_len = inputs.shape
        # In llama3, 'A' and ' A' are different tokens.
        choices = all_choices[:self.choice_cnt] + ['(' + item + ')' for item in all_choices[:self.choice_cnt]]
        # choices = ['A', 'B', 'C', 'D', 'E', '(A)', '(B)', '(C)', '(D)', '(E)']
        if self.args.model_name in ['llama3-8b-instruct', 'qwen2-7b-instruct']:
            choices = all_choices[:self.choice_cnt] + ['(' + item + ')' for item in all_choices[:self.choice_cnt]] + [' ' + item for item in all_choices[:self.choice_cnt]]
            # choices = ['A', 'B', 'C', 'D', 'E', '(A)', '(B)', '(C)', '(D)', '(E)', ' A', ' B', ' C', ' D', ' E']
        out_idx = [0 for _ in range(batch_size)]  # Default to the first token if nothing is found.
        seqs = outs['sequences']  # batch_size, seq_len, stores token IDs
        new_token_ids = seqs[:, input_len:]

        choices_idx = self.tokenizer(choices)['input_ids']
        if self.args.model_name in ['llama2-7b-chat', 'llama2-13b-chat']:
            # ['<s>', '_A'],  ['<s>', '(', 'A', ')']
            choices_idx = [item[1] if len(item) == 2 else item[2] for item in choices_idx]  # Token IDs for _A, A, etc.
            #['_A'], ['(A', ')']
        elif self.args.model_name in ['llama3-8b-instruct', 'qwen2-7b-instruct']:
            choices_idx = [item[0] for item in choices_idx]
        for bt in range(batch_size):  # Iterate over the batch.
            for idx in range(end_idx[bt]):  # Tokens in one sequence.
                token_id = new_token_ids[bt][idx]
                if token_id in choices_idx:  # First occurrence of the choice token.
                    out_idx[bt] = idx
                    break
        return out_idx, choices_idx      

    def get_need_idx_for_generation(self, probs, end_idx, mode):
        """
        Find the token indices to inspect based on the selected mode.
        Input:
            - mode: 
                - first, last, min, avg - get the index of the needed token
                - dim_min, dim_max - get the indices of all tokens, then take min/max over hidden dimensions
        """ 
        res_idx = []
        bt_size = probs.shape[0]
        text_len = probs.shape[1]
        assert mode in ['first', 'last', 'avg', 'min', 'dim_min', 'dim_max']
        if mode == 'first':
            res_idx = torch.zeros(bt_size, dtype=torch.int)     # Select the first position for all samples.
        elif mode == 'last':
            res_idx = [item if item != text_len else item - 1 for item in end_idx]  # Select the last position for all samples.
        elif mode == 'min':
            temp_idx = [item + 1 if item != text_len else item for item in end_idx]
            for bt in range(bt_size):
                min_prob, min_index = torch.min(probs[bt][:temp_idx[bt]], dim=-1)  # batch_size
                res_idx.append(min_index)
        elif mode == 'avg' or mode == 'dim_min' or mode == 'dim_max':
            for bt in range(bt_size):
                if end_idx[bt] == text_len:
                    res_idx.append(list(range(end_idx[bt])))
                else:
                    res_idx.append(list(range(end_idx[bt] + 1)))
        return res_idx
    
    def get_token_and_prob_for_each_pos(self, outs, bt_size, end_idx):
        """
        Get the top-1 token for each layer at every position (early exit) and the probability of the final generated token at each layer.
        """
        probs_for_generated_token = [[] for _ in range(bt_size)]  # Probability of the final generated token at each layer.
        tokens_for_each_pos = [[] for _ in range(bt_size)]
        for bt in range(bt_size):
            end_pos = end_idx[bt]
            for pos in range(end_pos):
                hidden_states_for_all_layers = []
                for layer in range(len(outs['hidden_states'][pos]))[1:]:
                    hidden_states = outs['hidden_states'][pos][layer][bt][-1]  # hidden_size
                    hidden_states_for_all_layers.append(hidden_states)
                hidden_states_for_all_layers = torch.stack(hidden_states_for_all_layers)  # (layers, hidden_dim)
                probs = nn.Softmax(dim=-1)(self.model.lm_head(hidden_states_for_all_layers))
                max_value_for_each_layer, max_token_for_each_layer = torch.max(probs, dim=-1)
                tokens_for_each_pos[bt].append(self.tokenizer.convert_ids_to_tokens(max_token_for_each_layer))
                generated_token = max_token_for_each_layer[-1]
                probs_for_generated_token[bt].append(probs[:, generated_token])
            
            probs_for_generated_token[bt] = torch.stack(probs_for_generated_token[bt]).t().tolist()
            probs_for_generated_token[bt] = [[round(element, 4) for element in row] for row in probs_for_generated_token[bt]]
            tokens_for_each_pos[bt] = [[tokens_for_each_pos[bt][j][i] for j in range(len(tokens_for_each_pos[bt]))] for i in range(len(tokens_for_each_pos[bt][0]))]
        return probs_for_generated_token, tokens_for_each_pos
    
    def get_generation_end(self, generated_tokens):
        # generated_tokens: batch_size, new_seq_len
        text_len = generated_tokens.shape[-1]
        end_idx = []
        for idx in range(len(generated_tokens)):
            eos_idx = torch.where(generated_tokens[idx] == self.eos_id_dict[self.args.model_name])[0]  # Returns a tuple; [0] is the tensor of matching positions.
            if len(eos_idx) == 0:  # No eos token found.
                end_idx.append(text_len)
            else:
                end_idx.append(eos_idx[0].item())  # First eos token position.
        return end_idx
    
    def get_generated_tokens_probs_entropy(self, scores, generated_tokens, bt_size):
        top_indices = []  # Store the token ID with the highest probability.
        top_scores = []  # Store the corresponding probabilities.
        ans_scores = []  # Store probabilities for the generated sequence.
        ans_entropy = []
        for idx in range(len(scores)):  # Iterate over each token.
            probs = nn.Softmax(dim=1)(scores[idx])  # batch_size, vocab_size
            tmp_scores, tmp_indices = torch.max(probs, dim=1)  # batch_size
            cur_scores = [probs[t, generated_tokens[t, idx]] for t in range(bt_size)]  # batch_size, probability of each generated token
            cur_entropy = torch.sum(-(probs * torch.log2(probs)), dim=1)  # batch_size

            # Only when greedy search is used, ans_scores equals top_scores.
            ans_scores.append(cur_scores)  # seq_len, batch_size
            ans_entropy.append(cur_entropy.tolist())
            top_indices.append(tmp_indices.tolist())
            top_scores.append(tmp_scores.tolist())
        
        top_indices = torch.tensor(top_indices, dtype=torch.int64).t()
        top_scores = torch.tensor(top_scores).t()  # batch_size, text_len
        ans_scores = torch.tensor(ans_scores).t()
        ans_entropy = torch.tensor(ans_entropy).t()
        return top_indices, top_scores, ans_scores, ans_entropy
    
    def get_confidence_idx(self, outs, inputs, end_idx):
        batch_size, input_len = inputs.shape
        seqs = outs['sequences']  # batch_size, seq_len, stores token IDs
        new_token_ids = seqs[:, input_len:]

        pattern = ['certain', 'uncertain', 'ġcertain', 'ġuncertain', '▁certain', '▁uncertain', '*certain', '*uncertain']
        res_idx = []
        for bt in range(len(new_token_ids)):
            bt_res = self.tokenizer.convert_ids_to_tokens(new_token_ids[bt][:end_idx[bt]])
            flag = 0
            for span_len in [3, 2, 1]:
                span_start = 0
                span_end = span_start + span_len - 1
                while span_end < len(bt_res):
                    span_text = ''.join(bt_res[span_start:span_end + 1]).lower().strip()
                    if span_text in pattern:
                        res_idx.append(span_start)
                        flag = 1
                        break   
                    span_start += 1
                    span_end += 1
                if flag == 1:
                    break
            if flag == 0:
                if self.args.model_name in ['llama2-7b-chat', 'llama2-13b-chat']:
                    res_idx.append(1)
                else:
                    res_idx.append(0)
        return res_idx
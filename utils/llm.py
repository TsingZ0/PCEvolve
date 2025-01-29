import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.getLogger('transformers').setLevel(logging.ERROR)


class LlamaWrapper(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        if args.server_llm == 'Llama2':
            model_dir = "llm/Llama-2-7b-chat-hf"
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir, 
                torch_dtype=torch.float16, 
                local_files_only=True,
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir, 
                torch_dtype=torch.float16, 
                local_files_only=True,
                device_map='auto'
            )
        elif args.server_llm == 'Llama3':
            model_dir = "llm/Meta-Llama-3-8B"
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir, 
                torch_dtype=torch.float16, 
                local_files_only=True,
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir, 
                torch_dtype=torch.float16, 
                local_files_only=True,
                device_map='auto'
            )
        else:
            raise NotImplementedError

        self.model = self.model.to(args.device)

    def __call__(self, prompt):
        with torch.no_grad():
            batch = self.tokenizer(
                prompt, 
                padding='max_length', 
                max_length=self.args.prompt_max_length, 
                truncation=True, 
                return_tensors="pt"
            ).to(self.args.device)
            out = self.model.generate(
                **batch, 
                max_new_tokens=self.args.prompt_max_length,
            )
            text = self.tokenizer.decode(
                out[0][len(sum(batch['attention_mask'])):], 
                skip_special_tokens=True,
            ).strip()
            return text
    

def get_llm(args):
    if 'Llama' in args.server_llm:
        return LlamaWrapper(args)
    else:
        raise NotImplementedError
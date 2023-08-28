import torch
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer
)
from auto_gptq import AutoGPTQForCausalLM
from peft import PeftModel


def load_automodel(model_name, trust_remote_code=False, dtype=torch.float16, load_in_8bit=False):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=dtype, trust_remote_code=trust_remote_code, load_in_8bit = load_in_8bit
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def load_llama_model(model_name):
    model = LlamaForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def load_peft_model(model_name, orig):
    model = LlamaForCausalLM.from_pretrained(
        model_name, device_map="auto"
    )
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = PeftModel.from_pretrained(model, orig)
    return model, tokenizer


def load_gptq_model(model_name, trust_remote_code=False, quantize_config=None,inject_fused_attention=True):
    model = AutoGPTQForCausalLM.from_quantized(
        model_name,
        use_safetensors=True,
        trust_remote_code=trust_remote_code,
        device="cuda:0",
        use_triton=False,
        quantize_config=quantize_config,
        inject_fused_attention=inject_fused_attention
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def load_from_catalogue(model_name):
    """
    A method to load the models of the shared task. A collection of instruction strings from the huggingface modelcards
    is provided with the dictionary as "user_prompt" and "assistant_prompt". These only are examples, you don't have to
    use them.
    @param model_name: The model to instanziate
    @return: model, tokenizer, user_prompt, assistant_prompt
    """
    catalogue = {
        "NousResearch/Nous-Hermes-13b": {
            "load_method": load_llama_model,
            "user_prompt": "### Instruction:",
            "assistant_prompt": "### Response:",
        },
        "TheBloke/guanaco-65B-GPTQ": {
            "load_method": lambda x: load_gptq_model(
                x, trust_remote_code=True, #might not need the trust remote code
            ),
            "user_prompt": "### Human:",
            "assistant_prompt": "### Assistant:",
        },
        "TheBloke/WizardLM-13B-V1.1-GPTQ": {
            "load_method": lambda x: load_gptq_model(
                x, trust_remote_code=True, #might not need the trust remote code
            ),
            "user_prompt": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n USER: ",
            "assistant_prompt": "Assistant: ",
        },
        "TheBloke/Platypus2-70B-Instruct-GPTQ": {
            "load_method": lambda x: load_gptq_model(
                x, trust_remote_code=False, inject_fused_attention=False
            ),
            "user_prompt": "### Instruction:",
            "assistant_prompt": "### Response:",
        },
        "Open-Orca/OpenOrca-Platypus2-13B":{
            "load_method": lambda x: load_automodel(x, trust_remote_code=True),
            "user_prompt": "### Instruction:",
            "assistant_prompt": "### Response:",
        },
          "psmathur/orca_mini_v3_7b":{
            "load_method": lambda x: load_automodel(x, trust_remote_code=True, load_in_8bit=True),
            "user_prompt": "### System: You are an AI assistant that follows instruction extremely well. Help as much as you can. \n\n ### User:",
            "assistant_prompt": "### Assistant:",
        }
    }


    model, tokenizer = catalogue[model_name]["load_method"](model_name)

    return model, tokenizer, catalogue[model_name]["user_prompt"], catalogue[model_name]["assistant_prompt"]

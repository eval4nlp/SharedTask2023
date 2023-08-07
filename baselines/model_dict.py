import torch
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from auto_gptq import AutoGPTQForCausalLM
from peft import PeftModel


def load_automodel(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.float16
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
        model_name, device_map="auto", torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = PeftModel.from_pretrained(model, orig)
    return model, tokenizer


def load_gptq_model(model_name, base_name, trust_remote_code=False, quantize_config=None,inject_fused_attention=True):
    model = AutoGPTQForCausalLM.from_quantized(
        model_name,
        model_basename=base_name,
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
                x, base_name="Guanaco-65B-GPTQ-4bit.act-order", trust_remote_code=True
            ),
            "user_prompt": "### Human:",
            "assistant_prompt": "### Assistant:",
        },
        "decapoda-research/llama-7b-hf": {
            "load_method": lambda x: load_peft_model(x, orig="avocardio/alpaca-lora-7b-german-base-52k"),
            "user_prompt": "",
            "assistant_prompt": "",
        }
    }


    model, tokenizer = catalogue[model_name]["load_method"](model_name)

    return model, tokenizer, catalogue[model_name]["user_prompt"], catalogue[model_name]["assistant_prompt"]

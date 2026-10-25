
# # import torch

# # from transformers import (
# #     AutoTokenizer,
# #     AutoModelForCausalLM,
# #     BitsAndBytesConfig
# # )


# # MODEL_NAME = "Qwen/Qwen3-8B"


# # print("Loading Qwen3-8B...")

# # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# # bnb_config = BitsAndBytesConfig(
# #     load_in_4bit=True,
# #     bnb_4bit_quant_type="nf4",
# #     bnb_4bit_compute_dtype=torch.float16,
# #     bnb_4bit_use_double_quant=True
# # )

# # model = AutoModelForCausalLM.from_pretrained(
# #     MODEL_NAME,
# #     quantization_config=bnb_config,
# #     device_map="auto"
# # )

# # print("Qwen3-8B loaded.")
# # ==============================================================================

# from langchain_ollama import ChatOllama


# llm = ChatOllama(
#     model="qwen3:8b",
#     temperature=0.7
# )


# # ==============================================================================

# def generate_8b(
#     prompt,
#     max_new_tokens=500,
#     do_sample=False,
#     temperature=0.7,
#     top_p=0.8,
#     top_k=20
# ):

#     messages = [
#         {
#             "role": "user",
#             "content": prompt
#         }
#     ]

#     model_inputs = tokenizer.apply_chat_template(
#         messages,
#         add_generation_prompt=True,
#         tokenize=True,
#         return_dict=True,
#         return_tensors="pt",
#         enable_thinking=False
#     ).to(model.device)

#     generation_kwargs = {
#         "max_new_tokens": max_new_tokens,
#         "do_sample": do_sample
#     }

#     if do_sample:
#         generation_kwargs.update({
#             "temperature": temperature,
#             "top_p": top_p,
#             "top_k": top_k
#         })

#     generated_ids = model.generate(
#         **model_inputs,
#         **generation_kwargs
#     )

#     input_length = model_inputs["input_ids"].shape[-1]

#     generated_ids = generated_ids[0][input_length:]

#     response = tokenizer.decode(
#         generated_ids,
#         skip_special_tokens=True
#     )

#     return response.strip()



from langchain_ollama import ChatOllama


def generate_8b(
    prompt,
    max_new_tokens=500,
    do_sample=False,
    temperature=0.7,
    top_p=0.8,
    top_k=20
):
    """
    Generate a response using local Ollama Qwen3-8B.

    Keeps the same function interface as the original
    Hugging Face implementation.
    """

    # Match the original HF behavior:
    # do_sample=False means deterministic generation.
    actual_temperature = temperature if do_sample else 0.0

    llm = ChatOllama(
        model="qwen3:8b",
        temperature=actual_temperature,
        top_p=top_p,
        top_k=top_k,
        num_predict=max_new_tokens
    )

    response = llm.invoke(
        prompt + "\n/no_think"
    )

    return response.content.strip()
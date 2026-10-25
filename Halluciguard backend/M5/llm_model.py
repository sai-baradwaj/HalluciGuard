# import os

# os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
# os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# from transformers import AutoTokenizer, AutoModelForCausalLM


# # ==========================================
# # LOAD MODEL ONCE
# # ==========================================

# model_name = "Qwen/Qwen3-0.6B"

# tokenizer = AutoTokenizer.from_pretrained(
#     model_name
# )

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto"
# )


# ==========================================
# GENERATE RESPONSE
# ==========================================

# def generate_response(
#     prompt,
#     max_new_tokens=500,
#     do_sample=True,
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


#     # --------------------------------------
#     # Generation settings
#     # --------------------------------------

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


#     # --------------------------------------
#     # Remove input tokens
#     # --------------------------------------

#     input_length = model_inputs["input_ids"].shape[-1]

#     generated_ids = generated_ids[0][input_length:]


#     # --------------------------------------
#     # Decode
#     # --------------------------------------

#     response = tokenizer.decode(
#         generated_ids,
#         skip_special_tokens=True
#     )

#     return response.strip()

import ollama


def generate_response(
    prompt,
    max_new_tokens=5000,
    do_sample=True,
    temperature=0.7,
    top_p=0.8,
    top_k=20
):
    response = ollama.chat(
        model="qwen3:8b",
        messages=[
            {
                "role": "user",
                "content": prompt.replace("\n/no_think", "")
            }
        ],
        think=False,
        options={
            "temperature": temperature if do_sample else 0.0,
            "top_p": top_p,
            "top_k": top_k,
            "num_predict": max_new_tokens
        }
    )

    return response.message.content.strip()

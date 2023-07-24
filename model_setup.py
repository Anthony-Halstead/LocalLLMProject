# import transformers
# import os
# import torch
# from dotenv import load_dotenv
# from torch import cuda
# import transformers


# torch.cuda.is_available()
# load_dotenv()

# # device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
# model_id = "meta-llama/Llama-2-7b-chat-hf"  


# # bnb_config = transformers.BitsAndBytesConfig(
# #     load_in_4bit = True,
# #     bnb_4bit_quant_type='nf4',
# #     bnb_4bit_use_double_quant=True,
# #     bnb_4bit_compute_dtype=bfloat16
# #                                              )

# hf_auth = os.getenv("HUGGING_FACE_KEY")
# model_config = transformers.AutoConfig.from_pretrained(
#     model_id,
#     use_auth_token=hf_auth)

# model = transformers.AutoModelForCausalLM.from_pretrained(
#     model_id,
#     trust_remote_code=True,
#     config=model_config,
#     device_map='auto',
#     use_auth_token=hf_auth
# )

# model.eval()
# # print(f"Model loaded on {device}")

# tokenizer = transformers.AutoTokenizer.from_pretrained(
#     model_id,
#     use_auth_token=hf_auth
# )

# stop_list = ['\nHuman', '\n```\n']

# # stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
# # stop_token_ids

# generate_text = transformers.pipeline(
#     model=model, tokenizer=tokenizer,
#     return_full_text=True, #Langchain expects full text
#     task='text-generation',
#     #model perameters can be passed here too
#     #stopping_criteria=stopping_criteria, #without this modelrambles during chat
#     temperature=0.0, # randomness of outputs, 0.0 is the min or less random 1.0 is max or more random/creative
#     max_new_tokens=512, # max number of tokens to generate in the output
#     repetition_penalty=1.1 # without this output begins repeating
# )

# res = generate_text("Explain to me the difference between nuclear fission and fusion.")
# print(res[0]["generated_text"])
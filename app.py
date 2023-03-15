from transformers import GPTJForCausalLM, GPT2Tokenizer
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Init is ran on server startup
# Load your model to GPU as a global variable here.
def init():
    global model
    global tokenizer

    print("loading to CPU...")
    model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    print("done")

    # conditionally load to GPU
    if device == "cuda:0":
        print("loading to GPU...")
        model.cuda()
        print("done")

    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-j-6B")


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global tokenizer

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    num_return_sequences = model_inputs.get('num_return_sequences', 1)
    max_length = model_inputs.get('max_length', 50)
    temperature = model_inputs.get('temperature', 1.0)
    top_k = model_inputs.get('top_k', 50)
    top_p = model_inputs.get('top_p', 1.0)
    repetition_penalty = model_inputs.get('repetition_penalty', 1.0)
    return_full_text = model_inputs.get('return_full_text', False)
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Tokenize inputs
    input_tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Run the model
    output = model.generate(input_tokens, max_length=max_length, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, do_sample=True, num_return_sequences=num_return_sequences, return_full_text=return_full_text)

    # Decode output tokens
    output_text = tokenizer.batch_decode(output, skip_special_tokens = True)[0]

    result = {"output": output_text}

    # Return the results as a dictionary
    return result

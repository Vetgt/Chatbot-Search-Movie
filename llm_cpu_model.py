from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2-1.5B-Instruct"

print(f"🔹 Loading {model_name} (optimized for CPU)...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = torch.device("cpu")
model = model.to(device)

def generate_response(prompt, max_new_tokens=100):
    messages = [
        {"role": "system", "content": "You are a helpful movie recommendation assistant."},
        {"role": "user", "content": prompt}
    ]
    formatted_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(formatted_input, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7, 
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response.strip()

if __name__ == "__main__":
    test_q = "Try to recommend me a good sci-fi movie from 2020."
    print(f"\nTest Input: {test_q}")
    print(f"Output: {generate_response(test_q)}")
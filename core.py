import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Hugging Face model_path
model_path = 'psmathur/orca_mini_3b'
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
model = model.to('mps')

class Instruction(BaseModel):
    instruction: str
    input: str = None

@app.post("/generate_text")
def generate_text(instruction: Instruction):
    system = 'You are an SAP consultant assistant that follows instructions extremely well. you chat with the user and answer as best as you can'

    if instruction.input:
        prompt = f"### System:\n{system}\n\n### User:\n{instruction.instruction}\n\n### Input:\n{instruction.input}\n\n### Response:\n"
    else:
        prompt = f"### System:\n{system}\n\n### User:\n{instruction.instruction}\n\n### Response:\n"

    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens).unsqueeze(0)
    tokens = tokens.to('mps')
    instance = {'input_ids': tokens, 'top_p': 1.0, 'temperature': 0.7, 'generate_len': 1024, 'top_k': 50}

    length = len(tokens[0])
    with torch.no_grad():
        rest = model.generate(
            input_ids=tokens.to('mps'),
            max_length=length + instance['generate_len'],
            use_cache=True,
            do_sample=True,
            top_p=instance['top_p'],
            temperature=instance['temperature'],
            top_k=instance['top_k']
        )
    output = rest[0][length:]
    res = tokenizer.decode(output, skip_special_tokens=True)
    return {"result": res}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

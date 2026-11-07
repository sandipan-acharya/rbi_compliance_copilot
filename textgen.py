from transformers import pipeline

_gen_pipe = None

def generate(text: str, mode: str = "summary", entities: list = None) -> str:
    global _gen_pipe
    if _gen_pipe is None:
        # Small instruction model that works on CPU with current transformers
        _gen_pipe = pipeline(
            "text-generation",
            model="HuggingFaceTB/SmolLM2-360M-Instruct",
            max_new_tokens=180,
            temperature=0.3,
            do_sample=True
        )
    
    ent_str = ""
    if entities:
        ent_str = "Key entities: " + ", ".join([f"{e['label']}:{e['span']}" for e in entities[:6]])

    prompts = {
        "summary": f"Summarise the following RBI clause in plain English (max 80 words):\n{text}\n{ent_str}\nSummary:",
        "checklist": f"Create a practical compliance checklist for bank staff from this clause:\n{text}\n{ent_str}\nChecklist:",
        "explain": f"Explain this RBI requirement simply for a compliance officer:\n{text}\nExplanation:"
    }
    prompt = prompts.get(mode, prompts["summary"])
    
    result = _gen_pipe(prompt)[0]["generated_text"]
    # Return only the generated part
    return result[len(prompt):].strip() if result.startswith(prompt) else result.strip()
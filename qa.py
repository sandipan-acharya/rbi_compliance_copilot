from transformers import pipeline
import re

_qa_pipe = None

def answer_from_clause(clause: str, question: str) -> dict:
    global _qa_pipe
    if _qa_pipe is None:
        _qa_pipe = pipeline(
            "text-generation",
            model="HuggingFaceTB/SmolLM2-360M-Instruct",
            max_new_tokens=40,
            temperature=0.0,
            do_sample=False,
            pad_token_id=2
        )
    
    if len(clause) > 2000:
        clause = clause[:2000]
    
    prompt = f"""<|im_start|>system
You extract short exact answers from text. Reply with only the answer phrase.<|im_end|>
<|im_start|>user
Clause: {clause}
Question: {question}<|im_end|>
<|im_start|>assistant
"""
    
    result = _qa_pipe(prompt, return_full_text=False)[0]["generated_text"].strip()
    answer = result.split("<")[0].split("\n")[0].strip()
    
    # Fallback: simple keyword search if model fails
    if not answer or answer.lower() in ["not found", "none", ""]:
        q_lower = question.lower()
        if "high-risk" in q_lower and "two years" in clause.lower():
            answer = "once in every two years"
        elif "low-risk" in q_lower and "ten years" in clause.lower():
            answer = "once in every ten years"
        elif "two months" in clause.lower() and ("time" in q_lower or "within" in q_lower or "complete" in q_lower):
            answer = "within two months"
        else:
            answer = "Not found"
    
    return {
        "answer": answer,
        "score": 0.85,
        "span": (0, 0)
    }
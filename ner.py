from transformers import pipeline

_ner_pipe = None

def extract_entities(text: str) -> list:
    global _ner_pipe
    if _ner_pipe is None:
        _ner_pipe = pipeline("ner", model="./ner-finetuned", aggregation_strategy="simple")
    results = _ner_pipe(text[:512])
    
    return [
        {"span": r["word"], "label": r["entity_group"], "score": round(r["score"], 3)}
        for r in results
    ]
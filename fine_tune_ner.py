# Part II – Fine-tune NER on RBI corpus
# 1. Sample ~600 sentences from data/corpus
# 2. Pre-label with regex (ACT, SECTION, AMOUNT ₹, DATE, KYC_DOC, etc.)
# 3. Hand-correct labels
# 4. Fine-tune bert-base-cased (or nlpueb/legal-bert-base-uncased)
# 5. Evaluate with seqeval vs dslim/bert-base-NER baseline
# 6. Save model and swap into ner.py

print("See Assignment Plan for full fine-tuning steps (M3).")
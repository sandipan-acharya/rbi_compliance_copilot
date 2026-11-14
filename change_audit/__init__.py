"""Change Validation Audit Copilot - Part IV (Situated Learning) workplace adaptation
of the RBI KYC & PSL Compliance Copilot (Parts I-III).

Mapping of Part I-III components to the audit workflow:
  NER (fine-tuned)        -> extract.py   : APP / VERSION / ENV / HOST / TIME / RESULT from evidence text
  Semantic similarity/QA  -> audit.py     : does evidence X support validation-plan step Y?
  RAG over the standard   -> standard.py  : retrieve clauses that justify each finding / answer auditor questions
  Text generation         -> recommend.py : recommendation note to the application team
"""

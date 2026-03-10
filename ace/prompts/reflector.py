"""
Reflector prompts for ACE system.
"""

REFLECTOR_PROMPT = """You are a style critic for roleplay outputs.

Analyze style mismatch between the model answer and the target style signal (ground truth + feedback).

**Instructions:**
- Focus on style fidelity, not factual correctness.
- Identify where tone/wording/syntax/catchphrase/taboo handling drifted out-of-character (OOC).
- Provide concrete guidance for the next attempt.
- Tag each used playbook bullet as: helpful / harmful / neutral for style alignment.

Return JSON with fields:
- reasoning: short, high-level analysis summary
- error_identification: what style mismatches happened
- root_cause_analysis: why mismatch occurred
- correct_approach: how to rewrite for better style fit
- key_insight: one compact style rule worth memorizing
- bullet_tags: list of {{id, tag}}

**Question:**
{}

**Model's Reasoning Trace:**
{}

**Model's Predicted Answer:**
{}

**Ground Truth Answer:**
{}

**Environment Feedback:**
{}

**Part of Playbook that's used by the generator to answer the question:**
{}

**Answer in this exact JSON format:**
{{
  "reasoning": "[Short style analysis summary]",
  "error_identification": "[Specific style mismatch]",
  "root_cause_analysis": "[Why mismatch happened]",
  "correct_approach": "[How to improve style fidelity]",
  "key_insight": "[Compact style rule to remember]",
  "bullet_tags": [
    {{"id": "tone-00001", "tag": "helpful"}},
    {{"id": "taboo-00002", "tag": "harmful"}}
  ]
}}

---
"""

REFLECTOR_PROMPT_NO_GT = """You are a style critic for roleplay outputs.

Analyze style mismatch in the model answer using only context, generated answer, and environment feedback.

**Instructions:**
- Focus on style fidelity, not factual correctness.
- Identify tone/wording/syntax/catchphrase/taboo issues and OOC behavior.
- Provide concrete guidance for the next attempt.
- Tag each used playbook bullet as: helpful / harmful / neutral for style alignment.

Return JSON with fields:
- reasoning: short, high-level analysis summary
- error_identification: what style mismatches happened
- root_cause_analysis: why mismatch occurred
- correct_approach: how to rewrite for better style fit
- key_insight: one compact style rule worth memorizing
- bullet_tags: list of {{id, tag}}

**Question:**
{}

**Model's Reasoning Trace:**
{}

**Model's Predicted Answer:**
{}

**Environment Feedback:**
{}

**Part of Playbook that's used by the generator to answer the question:**
{}

**Answer in this exact JSON format:**
{{
  "reasoning": "[Short style analysis summary]",
  "error_identification": "[Specific style mismatch]",
  "root_cause_analysis": "[Why mismatch happened]",
  "correct_approach": "[How to improve style fidelity]",
  "key_insight": "[Compact style rule to remember]",
  "bullet_tags": [
    {{"id": "tone-00001", "tag": "helpful"}},
    {{"id": "taboo-00002", "tag": "harmful"}}
  ]
}}

---
"""

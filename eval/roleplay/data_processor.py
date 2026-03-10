import os
import json
import csv
import random
from typing import List, Dict, Any, Tuple
from difflib import SequenceMatcher

import openai


STYLE_CRITIC_PROMPT = """You are a strict roleplay style critic.

Score the predicted reply against the target role style reference.

Scoring rubric (1-10):
- 9-10: excellent in-character style match
- 7-8: mostly in-character, minor style drift
- 5-6: partially in-character, notable mismatch
- 1-4: largely out-of-character

Return JSON with:
- score: integer 1-10
- rationale: concise explanation of mismatch/alignment

Context:
{context}

Prompt:
{question}

Predicted reply:
{predicted}

Reference style reply:
{ground_truth}
"""


def load_data(data_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = []
    if data_path.lower().endswith('.csv'):
        with open(data_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = [row for row in reader]
    else:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

    print(f"Loaded {len(data)} samples from {data_path}")
    return data


class DataProcessor:
    """Roleplay data processor with LLM-based style scoring."""

    def __init__(self, task_name: str, critic_model: str = "gpt-4o", pass_threshold: int = 8):
        self.task_name = task_name
        self.critic_model = critic_model
        self.pass_threshold = pass_threshold
        self._last_feedback = ""
        self._score_cache: Dict[Tuple[str, str], Tuple[int, str]] = {}

        api_key = os.getenv("OPENAI_API_KEY", "")
        self.critic_client = openai.OpenAI(api_key=api_key) if api_key else None

    @staticmethod
    def create_roleplay_samples_from_dialog_csv(
        rows: List[Dict[str, Any]],
        focus_role: str = "hiro",
        context_turn_window: int = 6,
        min_context_chars: int = 1,
    ) -> List[Dict[str, Any]]:
        """Convert dialogue CSV rows into roleplay-style training samples.

        Expected CSV columns include:
        - translation: utterance text
        - reol: speaker/role name
        - optional metadata such as Act/Chapter/type/code
        """
        dialogue_rows: List[Dict[str, str]] = []
        for row in rows:
            text = (row.get("translation") or "").strip()
            role = (row.get("reol") or "").strip()
            if not text or not role:
                continue
            dialogue_rows.append({
                "role": role,
                "text": text,
                "act": (row.get("Act") or "").strip(),
                "chapter": (row.get("Chapter") or "").strip(),
                "type": (row.get("type") or "").strip(),
                "code": (row.get("code") or "").strip(),
            })

        focus_role_lower = focus_role.strip().lower()
        processed: List[Dict[str, Any]] = []

        for idx, row in enumerate(dialogue_rows):
            if row["role"].lower() != focus_role_lower:
                continue

            context_start = max(0, idx - context_turn_window)
            context_turns = dialogue_rows[context_start:idx]
            if not context_turns:
                continue

            context = "\n".join([f"{turn['role']}: {turn['text']}" for turn in context_turns]).strip()
            if len(context) < min_context_chars:
                continue

            speaker_name = focus_role.capitalize()
            question = (
                f"请你继续下面对话，并严格扮演{speaker_name}，"
                f"保持其语气、措辞和人物风格。只输出{speaker_name}下一句台词。"
            )

            style_refs = [
                turn["text"] for turn in dialogue_rows[max(0, idx - 80): idx]
                if turn["role"].lower() == focus_role_lower
            ]
            reference_style = "\n".join(style_refs[-8:])

            processed.append({
                "context": context,
                "question": question,
                "target": row["text"],
                "reference_style": reference_style,
                "others": {
                    "task": "roleplay",
                    "source": "dialog_csv",
                    "focus_role": focus_role,
                    "act": row["act"],
                    "chapter": row["chapter"],
                    "type": row["type"],
                    "code": row["code"],
                },
            })

        return processed

    @staticmethod
    def split_samples(
        samples: List[Dict[str, Any]],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        shuffled = list(samples)
        random.Random(seed).shuffle(shuffled)

        train_end = int(len(shuffled) * train_ratio)
        val_end = train_end + int(len(shuffled) * val_ratio)
        return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]

    def process_task_data(self, raw_data: List[Dict]) -> List[Dict]:
        if raw_data and "translation" in raw_data[0] and "reol" in raw_data[0]:
            return self.create_roleplay_samples_from_dialog_csv(raw_data)

        processed_data = []
        for item in raw_data:
            processed_data.append({
                "context": item.get("context", ""),
                "question": item.get("question", ""),
                "target": item.get("target", ""),
                "others": {
                    "task": self.task_name,
                    "reference_style": item.get("reference_style", "")
                }
            })
        return processed_data

    def _fallback_style_score(self, predicted: str, ground_truth: str) -> Tuple[int, str]:
        """Fallback scorer used when API key or critic call is unavailable."""
        ratio = SequenceMatcher(None, predicted.strip().lower(), ground_truth.strip().lower()).ratio()
        score = max(1, min(10, int(round(ratio * 10))))
        return score, f"Fallback lexical similarity score (ratio={ratio:.2f})"

    def _get_or_score(self, predicted: str, ground_truth: str) -> Tuple[int, str]:
        """Get style score from cache or compute it once."""
        cache_key = (predicted, ground_truth)
        if cache_key in self._score_cache:
            return self._score_cache[cache_key]

        score, rationale = self._score_style(predicted, ground_truth)
        self._score_cache[cache_key] = (score, rationale)
        return score, rationale

    def _score_style(self, predicted: str, ground_truth: str, context: str = "", question: str = "") -> Tuple[int, str]:
        prompt = STYLE_CRITIC_PROMPT.format(
            context=context,
            question=question,
            predicted=predicted,
            ground_truth=ground_truth
        )

        if self.critic_client is None:
            return self._fallback_style_score(predicted, ground_truth)

        try:
            response = self.critic_client.chat.completions.create(
                model=self.critic_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content or "{}"
            parsed = json.loads(content)

            score = int(parsed.get("score", 1))
            score = max(1, min(10, score))
            rationale = str(parsed.get("rationale", "No rationale provided"))
            return score, rationale
        except Exception as exc:
            score, rationale = self._fallback_style_score(predicted, ground_truth)
            return score, f"{rationale}; critic_call_error={type(exc).__name__}"

    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        score, rationale = self._get_or_score(predicted, ground_truth)
        self._last_feedback = f"Style score={score}/10. {rationale}"
        return score >= self.pass_threshold

    def evaluate_accuracy(self, out: List[str], target: List[str]) -> float:
        if len(out) != len(target):
            raise ValueError("Predictions and ground truths must have the same length.")

        passed = 0
        total_score = 0
        for predicted, ground_truth in zip(out, target):
            score, _ = self._get_or_score(predicted, ground_truth)
            total_score += score
            if score >= self.pass_threshold:
                passed += 1

        n = len(out) if out else 1
        print(f"  Style pass@{self.pass_threshold}: {passed}/{n} = {passed / n:.3f}")
        print(f"  Avg style score: {total_score / n:.2f}/10")
        return passed / n

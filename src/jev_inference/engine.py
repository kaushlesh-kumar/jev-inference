import math
import string
import threading
from time import perf_counter

from .backends.base import create_backend
from .prompts import label_prompt
from .schemas import ChoiceScore, DecisionRequest, DecisionResponse


def _value(value) -> float:
    item = getattr(value, "item", None)
    return float(item() if item is not None else value)


def normalize(scores: list[float], temperature: float) -> list[float]:
    if not scores or not all(math.isfinite(value) for value in scores):
        raise ValueError("model returned non-finite or empty scores")
    maximum = max(scores)
    weights = [math.exp((value - maximum) / temperature) for value in scores]
    total = sum(weights)
    return [value / total for value in weights]


class DecisionEngine:
    def __init__(
        self,
        model: str,
        device: str = "auto",
        max_input_tokens: int = 4096,
        dtype: str = "auto",
        backend: str = "transformers",
        backend_instance=None,
        **backend_options,
    ):
        if backend_instance is not None:
            runtime = backend_instance
        elif isinstance(backend, str):
            runtime = create_backend(
                backend,
                model,
                device=device,
                dtype=dtype,
                max_input_tokens=max_input_tokens,
                **backend_options,
            )
        else:
            runtime = backend
        self.backend = runtime
        self.model_id = runtime.model_id
        self.max_input_tokens = min(max_input_tokens, runtime.max_context_tokens)
        self._lock = threading.Lock()

    def _encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return self.backend.encode(text, add_special_tokens=add_special_tokens)

    def _special_token_ids(self) -> set[int]:
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is not None:
            return set(tokenizer.all_special_ids)
        return self.backend.special_token_ids()

    def _token_ids(self, labels: list[str]) -> list[int]:
        ids = [self._encode(label, add_special_tokens=False) for label in labels]
        if any(len(tokens) != 1 for tokens in ids):
            raise ValueError("this tokenizer does not support the required single-token labels")
        result = [tokens[0] for tokens in ids]
        if len(set(result)) != len(result) or any(
            token in self._special_token_ids() for token in result
        ):
            raise ValueError("choice labels must map to distinct, non-special tokens")
        return result

    def _inputs(self, prompt: str):
        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")
        if inputs["input_ids"].shape[1] > self.max_input_tokens:
            raise ValueError(f"prompt exceeds {self.max_input_tokens} tokens; reduce the input")
        return inputs.to(self.model.device)

    def _logits(self, prompt: str):
        backend = getattr(self, "backend", None)
        if backend is not None:
            return backend.next_token_logits(prompt)
        inputs = self._inputs(prompt)
        with self.torch.inference_mode():
            output = self.model(**inputs, use_cache=False)
        return output.logits[0, -1].float()

    def _sequence_score(self, prompt: str, choice: str) -> float:
        backend = getattr(self, "backend", None)
        if backend is not None:
            return backend.sequence_logprob(prompt, choice)
        inputs = self._inputs(prompt)
        prefix_length = inputs["input_ids"].shape[1]
        target_ids = self.tokenizer.encode(choice, add_special_tokens=False)
        if not target_ids:
            raise ValueError("choice must tokenize to at least one token")
        if prefix_length + len(target_ids) > self.max_input_tokens:
            raise ValueError(f"prompt and choice exceed {self.max_input_tokens} tokens")
        targets = self.torch.tensor([target_ids], device=self.model.device)
        input_ids = self.torch.cat([inputs["input_ids"], targets], dim=1)
        attention_mask = self.torch.ones_like(input_ids)
        with self.torch.inference_mode():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = output.logits[0, prefix_length - 1 : -1].float()
            log_probs = logits.log_softmax(dim=-1)
            selected = log_probs.gather(1, targets[0].unsqueeze(1))
            return selected.mean().item()

    def _scores(self, request: DecisionRequest) -> list[float]:
        if request.mode == "sequence_likelihood":
            prompt = (
                "Choose the best option for the context below. Reply with the exact option text "
                "and no explanation.\n\n"
                f"Context:\n{request.context}\n\nAllowed choices:\n"
                + "\n".join(request.choices)
                + "\n\nAnswer:"
            )
            return [self._sequence_score(prompt, choice) for choice in request.choices]

        if request.mode == "labels":
            labels = list(string.ascii_uppercase[: len(request.choices)])
            token_ids = self._token_ids(labels)
            prompt = label_prompt(request.context, request.choices)
            logits = self._logits(prompt)
            return [_value(logits[token]) for token in token_ids]
        yes_id, no_id = self._token_ids(["Yes", "No"])
        scores = []
        for choice in request.choices:
            prompt = (
                "Decide whether the proposed choice is appropriate for the context. "
                "Reply with exactly Yes or No and no explanation.\n\n"
                f"Context:\n{request.context}\n\nProposed choice:\n{choice}\n\nAnswer:"
            )
            logits = self._logits(prompt)
            scores.append(_value(logits[yes_id] - logits[no_id]))
        return scores

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        with self._lock:
            started = perf_counter()
            scores = self._scores(request)
            probabilities = normalize(scores, request.temperature)
            index = max(range(len(scores)), key=scores.__getitem__)
            return DecisionResponse(
                model=self.model_id,
                mode=request.mode,
                choice=request.choices[index],
                index=index,
                choices=[
                    ChoiceScore(choice=choice, score=score, probability=probability)
                    for choice, score, probability in zip(request.choices, scores, probabilities)
                ],
                latency_ms=(perf_counter() - started) * 1000,
                forward_passes=1 if request.mode == "labels" else len(request.choices),
            )

    def generate(self, prompt: str, max_new_tokens: int) -> tuple[str, int, bool]:
        return self.backend.generate(prompt, max_new_tokens)

    def synchronize(self) -> None:
        self.backend.synchronize()

    def memory_allocated(self) -> float | None:
        return self.backend.memory_allocated()

    def reset_peak_memory(self) -> None:
        self.backend.reset_peak_memory()

    def max_memory_allocated(self) -> float | None:
        return self.backend.max_memory_allocated()

    def max_memory_reserved(self) -> float | None:
        return self.backend.max_memory_reserved()

    def device_properties(self) -> dict:
        return self.backend.device_properties()

    def runtime_metadata(self) -> dict:
        return self.backend.runtime_metadata()

    def close(self) -> None:
        self.backend.close()

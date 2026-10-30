"""Hugging Face Transformers backend."""

from __future__ import annotations

from time import perf_counter


class TransformersBackend:
    backend_name = "transformers"

    def __init__(
        self, model: str, device: str = "auto", max_input_tokens: int = 4096, dtype: str = "auto"
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if dtype not in {"auto", "float16", "bfloat16", "float32"}:
            raise ValueError("unsupported model dtype")
        self.model_id = model
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map=device,
            torch_dtype="auto" if dtype == "auto" else getattr(torch, dtype),
            trust_remote_code=False,
        ).eval()
        context_limit = getattr(self.model.config, "max_position_embeddings", max_input_tokens)
        self.max_context_tokens = min(max_input_tokens, context_limit)
        self.is_cuda = self._check_cuda()

    def _check_cuda(self) -> bool:
        try:
            device = self.model.device
            if getattr(device, "type", None) != "cuda":
                return False
            return {parameter.device for parameter in self.model.parameters()} == {device}
        except (AttributeError, TypeError):
            return False

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def special_token_ids(self) -> set[int]:
        return set(self.tokenizer.all_special_ids)

    def format_prompt(self, prompt: str) -> str:
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        return prompt

    def _inputs(self, prompt: str):
        formatted = self.format_prompt(prompt)
        if self.tokenizer.chat_template:
            inputs = self.tokenizer(formatted, return_tensors="pt", add_special_tokens=False)
        else:
            inputs = self.tokenizer(formatted, return_tensors="pt")
        if inputs["input_ids"].shape[1] > self.max_context_tokens:
            raise ValueError(f"prompt exceeds {self.max_context_tokens} tokens; reduce the input")
        return inputs.to(self.model.device)

    def next_token_logits(self, prompt: str) -> list[float]:
        inputs = self._inputs(prompt)
        with self.torch.inference_mode():
            output = self.model(**inputs, use_cache=False)
        return output.logits[0, -1].float().tolist()

    def sequence_logprob(self, prompt: str, choice: str) -> float:
        inputs = self._inputs(prompt)
        prefix_length = inputs["input_ids"].shape[1]
        target_ids = self.encode(choice)
        if not target_ids:
            raise ValueError("choice must tokenize to at least one token")
        if prefix_length + len(target_ids) > self.max_context_tokens:
            raise ValueError(f"prompt and choice exceed {self.max_context_tokens} tokens")
        targets = self.torch.tensor([target_ids], device=self.model.device)
        input_ids = self.torch.cat([inputs["input_ids"], targets], dim=1)
        attention_mask = self.torch.ones_like(input_ids)
        with self.torch.inference_mode():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = output.logits[0, prefix_length - 1 : -1].float()
            log_probs = logits.log_softmax(dim=-1)
            selected = log_probs.gather(1, targets[0].unsqueeze(1))
            return selected.mean().item()

    def generate(self, prompt: str, max_new_tokens: int) -> tuple[str, int, bool]:
        from transformers import GenerationConfig

        inputs = self._inputs(prompt)
        eos = self.model.generation_config.eos_token_id
        config = GenerationConfig(
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            eos_token_id=eos,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, generation_config=config)
        tokens = output[0, inputs["input_ids"].shape[1] :].tolist()
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        eos_ids = eos if isinstance(eos, list) else [eos]
        hit_limit = len(tokens) >= max_new_tokens and tokens[-1] not in eos_ids
        return text, len(tokens), hit_limit

    def close(self) -> None:
        self.model = None
        if self.is_cuda:
            self.torch.cuda.empty_cache()

    def synchronize(self) -> None:
        if self.is_cuda:
            self.torch.cuda.synchronize(self.model.device)

    def memory_allocated(self) -> float | None:
        if not self.is_cuda:
            return None
        return self.torch.cuda.memory_allocated(self.model.device)

    def reset_peak_memory(self) -> None:
        if self.is_cuda:
            self.torch.cuda.reset_peak_memory_stats(self.model.device)

    def max_memory_allocated(self) -> float | None:
        if not self.is_cuda:
            return None
        return self.torch.cuda.max_memory_allocated(self.model.device)

    def max_memory_reserved(self) -> float | None:
        if not self.is_cuda:
            return None
        return self.torch.cuda.max_memory_reserved(self.model.device)

    def device_properties(self) -> dict:
        if not self.is_cuda:
            return {
                "name": str(self.model.device),
                "total_memory": None,
                "compute_capability": None,
            }
        properties = self.torch.cuda.get_device_properties(self.model.device)
        return {
            "name": properties.name,
            "total_memory": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
        }

    def runtime_metadata(self) -> dict:
        import torch
        import transformers

        return {
            "backend": self.backend_name,
            "backend_version": transformers.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "quantization": None,
            "dtype": str(self.model.dtype),
            "model_revision": getattr(self.model.config, "_commit_hash", None),
        }


def timed_forward(backend, operation):
    """Small helper used by benchmarks when a backend needs explicit sync timing."""
    backend.synchronize()
    started = perf_counter()
    result = operation()
    backend.synchronize()
    return result, (perf_counter() - started) * 1000

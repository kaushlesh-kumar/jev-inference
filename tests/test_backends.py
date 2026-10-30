import math

import pytest

from jev_inference.backends.base import Backend
from jev_inference.backends.llamacpp_backend import LlamaCppBackend
from jev_inference.engine import DecisionEngine
from jev_inference.schemas import DecisionRequest


class FakeBackend:
    backend_name = "fake"
    model_id = "fake-model"
    max_context_tokens = 64
    is_cuda = True

    def __init__(self):
        self.calls = []
        self.logits = [0.0] * 32
        self.encode_map = {"A": [0], "B": [1], "C": [2], "Yes": [3], "No": [4]}
        self.sequence = -1.0

    def encode(self, text, add_special_tokens=False):
        return list(self.encode_map.get(text, [10]))

    def special_token_ids(self):
        return {31}

    def format_prompt(self, prompt):
        return prompt

    def next_token_logits(self, prompt):
        self.calls.append(("logits", prompt))
        return list(self.logits)

    def sequence_logprob(self, prompt, choice):
        self.calls.append(("sequence", prompt, choice))
        return self.sequence

    def generate(self, prompt, max_new_tokens):
        self.calls.append(("generate", prompt, max_new_tokens))
        return "X", 1, False

    def synchronize(self):
        pass

    def memory_allocated(self):
        return 0.0

    def reset_peak_memory(self):
        pass

    def max_memory_allocated(self):
        return 0.0

    def max_memory_reserved(self):
        return 0.0

    def device_properties(self):
        return {"name": "fake"}

    def runtime_metadata(self):
        return {"backend": "fake", "backend_version": "0", "quantization": None}

    def close(self):
        pass


def engine_for(backend):
    return DecisionEngine("fake", backend_instance=backend)


def test_backend_protocol_is_runtime_checkable():
    assert isinstance(FakeBackend(), Backend)


def test_engine_labels_uses_backend_next_token_logits():
    backend = FakeBackend()
    backend.logits[0], backend.logits[1], backend.logits[2] = 1.0, 5.0, 3.0
    result = engine_for(backend).decide(
        DecisionRequest(context="state", choices=["first", "second", "third"], mode="labels")
    )
    assert result.choice == "second"
    assert result.index == 1
    assert result.forward_passes == 1
    assert [item.score for item in result.choices] == [1.0, 5.0, 3.0]
    assert sum(item.probability for item in result.choices) == pytest.approx(1)


def test_engine_independent_uses_yes_minus_no_per_option():
    backend = FakeBackend()

    def logits(prompt):
        vector = [0.0] * 32
        vector[3] = 10.0 if "good" in prompt else 1.0
        vector[4] = 0.0
        return vector

    backend.next_token_logits = logits
    result = engine_for(backend).decide(
        DecisionRequest(context="state", choices=["good", "bad"], mode="independent")
    )
    assert result.choice == "good"
    assert result.forward_passes == 2
    assert [item.score for item in result.choices] == [10.0, 1.0]


def test_engine_sequence_likelihood_delegates_to_backend():
    backend = FakeBackend()
    backend.sequence = -2.0
    result = engine_for(backend).decide(
        DecisionRequest(context="state", choices=["alpha", "beta"], mode="sequence_likelihood")
    )
    assert result.choice == "alpha"
    assert result.forward_passes == 2
    sequence_calls = [call for call in backend.calls if call[0] == "sequence"]
    assert [call[2] for call in sequence_calls] == ["alpha", "beta"]


@pytest.mark.parametrize(
    "encode_map",
    [
        {"A": [0, 1], "B": [2]},
        {"A": [0], "B": [0]},
        {"A": [31], "B": [1]},
    ],
)
def test_engine_rejects_labels_that_are_not_single_distinct_non_special_tokens(encode_map):
    backend = FakeBackend()
    backend.encode_map = encode_map
    with pytest.raises(ValueError, match="label|token"):
        engine_for(backend).decide(
            DecisionRequest(context="state", choices=["one", "two"], mode="labels")
        )


def test_engine_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unsupported backend"):
        DecisionEngine("x", backend="does-not-exist")


def test_engine_exposes_measurement_and_generation_hooks():
    backend = FakeBackend()
    engine = engine_for(backend)
    assert engine.generate("prompt", 8) == ("X", 1, False)
    assert engine.memory_allocated() == 0.0
    assert engine.runtime_metadata()["backend"] == "fake"


class FakeLlama:
    metadata = {"tokenizer.chat_template": "template"}

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.tokens = []
        self.reset_count = 0
        self.eval_calls = []
        self._last = 0

    def n_ctx(self):
        return 64

    def n_vocab(self):
        return 4

    def token_bos(self):
        return 3

    def token_eos(self):
        return 2

    def tokenize(self, text, add_bos=True, special=False):
        assert add_bos is False
        return {
            "prompt": [0],
            "<user>prompt</user><assistant>": [0],
            "ab": [1, 2],
            "x": [1],
        }[text.decode()]

    def reset(self):
        self.reset_count += 1
        self.tokens.clear()

    def eval(self, tokens):
        self.eval_calls.append(list(tokens))
        self.tokens.extend(tokens)
        self._last = tokens[-1]

    @property
    def logits(self):
        return [0.0] * 4

    def create_completion(self, prompt, **kwargs):
        return {
            "choices": [{"text": "X", "finish_reason": "length"}],
            "usage": {"completion_tokens": 2},
        }


def llama_backend(**kwargs):
    return LlamaCppBackend(
        "fake.gguf",
        max_input_tokens=64,
        llama_factory=FakeLlama,
        chat_apply=lambda template, messages, add_assistant: (
            f"<user>{messages[0]['content']}</user><assistant>"
        ),
        **kwargs,
    )


def test_llamacpp_scores_next_token_without_bos_and_resets_state():
    backend = llama_backend()
    logits = backend.next_token_logits("prompt")
    assert logits == [0.0] * 4
    assert backend._llama.reset_count == 1
    assert backend._llama.eval_calls == [[0]]
    assert backend._llama.kwargs["logits_all"] is False


def test_llamacpp_sequence_logprob_teacher_forces_each_target_token():
    backend = llama_backend()
    score = backend.sequence_logprob("prompt", "ab")
    assert score == pytest.approx(-math.log(4))
    assert backend._llama.reset_count == 1
    assert backend._llama.eval_calls == [[0], [1], [2]]


def test_llamacpp_sequence_rejects_context_overflow_before_eval():
    backend = llama_backend()
    backend.max_context_tokens = 2
    with pytest.raises(ValueError, match="exceed"):
        backend.sequence_logprob("prompt", "ab")


def test_llamacpp_formats_prompt_with_chat_template():
    backend = llama_backend()
    assert backend.format_prompt("hello") == "<user>hello</user><assistant>"
    assert backend.runtime_metadata()["quantization"] is None


def test_llamacpp_generation_reports_length_limit():
    backend = llama_backend()
    text, tokens, hit = backend.generate("prompt", 8)
    assert (text, tokens, hit) == ("X", 2, True)

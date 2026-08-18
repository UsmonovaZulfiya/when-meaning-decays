import os
import torch
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    # This fallback is required by some Gemma 3 checkpoints and Transformers versions.
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None


BACKEND_LABEL = "full precision"


@dataclass
class SimpleResponse:
    content: str


class LocalChatModel:
    """
    Local chat wrapper for original/full-precision model loading.

    This version does NOT use bitsandbytes and does NOT apply 4-bit quantization.

    Supports:
      model.invoke([HumanMessage("...")]).content
      model.invoke_batch(["prompt 1", "prompt 2", ...])
    """

    def __init__(
        self,
        model_path: str,
        max_batch_size: int = 8,
        default_temperature: float = 0.7,
        default_max_new_tokens: int = 140,
        top_p: float = 0.95,
        repetition_penalty: float = 1.05,
        torch_dtype=torch.bfloat16,
    ):
        self.model_path = model_path
        self.max_batch_size = max_batch_size
        self.default_temperature = default_temperature
        self.default_max_new_tokens = default_max_new_tokens
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.torch_dtype = torch_dtype

        if not os.path.isdir(model_path):
            raise RuntimeError(f"Model path does not exist or is not a directory: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=True,
            local_files_only=True,
        )

        # Left padding is required for decoder-only batched generation.
        self.tokenizer.padding_side = "left"

        # A padding token is required for batched generation.
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        model_kwargs = {
            "device_map": {"": 0},
            "dtype": self.torch_dtype,
            "trust_remote_code": True,
            "local_files_only": True,
            "low_cpu_mem_usage": True,
        }

        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        except Exception as causal_lm_error:
            if AutoModelForImageTextToText is None:
                raise causal_lm_error

            print(
                "AutoModelForCausalLM loading failed; trying "
                "AutoModelForImageTextToText instead."
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                **model_kwargs,
            )
        print("CUDA available:", torch.cuda.is_available())
        print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
        print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")
        print("Model device:", self.model.device)
        print("HF device map:", getattr(self.model, "hf_device_map", None))
        print("First param device:", next(self.model.parameters()).device)

        self.model.eval()

    def _build_prompt(self, user_text: str) -> str:
        """
        Builds model-specific chat prompt.
        Llama/Gemma usually have chat templates.
        """
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": user_text}]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return f"User: {user_text}\nAssistant:"

    @torch.inference_mode()
    def invoke(self, messages, max_new_tokens: int | None = None, temperature: float | None = None):
        m0 = messages[0]
        user_text = m0.content if hasattr(m0, "content") else str(m0)

        responses = self.invoke_batch(
            [user_text],
            batch_size=1,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        return responses[0]

    @torch.inference_mode()
    def invoke_batch(
        self,
        prompts,
        batch_size: int | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ):
        """
        Batched generation.

        Args:
            prompts: list[str]
            batch_size: number of prompts processed together
            max_new_tokens: optional override for generation length
            temperature: optional override for generation temperature
        Returns:
            list[SimpleResponse]
        """

        if batch_size is None:
            batch_size = self.max_batch_size

        if max_new_tokens is None:
            max_new_tokens = self.default_max_new_tokens

        all_outputs = []
        effective_temperature = self.default_temperature if temperature is None else temperature
        do_sample = effective_temperature > 0

        for start in range(0, len(prompts), batch_size):
            batch_user_prompts = prompts[start:start + batch_size]
            batch_chat_prompts = [self._build_prompt(p) for p in batch_user_prompts]

            inputs = self.tokenizer(
                batch_chat_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.model.device)

            input_len = inputs["input_ids"].shape[1]

            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "repetition_penalty": self.repetition_penalty,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "remove_invalid_values": True,
                "renormalize_logits": True,
            }

            if do_sample:
                generation_kwargs["temperature"] = effective_temperature
                generation_kwargs["top_p"] = self.top_p

            out = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

            for i in range(out.shape[0]):
                generated_tokens = out[i, input_len:]

                content = self.tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()

                all_outputs.append(SimpleResponse(content=content))

        return all_outputs

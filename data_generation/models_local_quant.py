import torch
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

try:
    # This fallback is required by some Gemma 3 checkpoints and Transformers versions.
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None


BACKEND_LABEL = "4-bit quantized"


@dataclass
class SimpleResponse:
    content: str


class LocalChatModel:
    """
    Local OpenAI-chat-like wrapper for decoder-only HF models.

    Uses 4-bit bitsandbytes NF4 quantization:
      - load_in_4bit=True
      - bnb_4bit_quant_type="nf4"
      - bnb_4bit_use_double_quant=True
      - bnb_4bit_compute_dtype=torch.bfloat16

    The experiment calls invoke_batch(...) separately for:
      - guessing, usually temperature=0.0
      - description generation, usually temperature=0.8
    """

    def __init__(
        self,
        model_path: str,
        max_batch_size: int = 8,
        default_temperature: float = 0.7,
        default_max_new_tokens: int = 140,
        top_p: float = 0.95,
        repetition_penalty: float = 1.05,
    ):
        self.model_path = model_path
        self.max_batch_size = max_batch_size
        self.default_temperature = default_temperature
        self.default_max_new_tokens = default_max_new_tokens
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Run this on a GPU node.")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=True,
            local_files_only=True,
        )

        # Left padding is required for decoder-only batched generation.
        self.tokenizer.padding_side = "left"

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map={"": "cuda:0"},
                trust_remote_code=True,
                local_files_only=True,
                low_cpu_mem_usage=True,
                dtype=torch.bfloat16,
            )
        except Exception as causal_lm_error:
            # Some Gemma 3 checkpoints are registered as image-text-to-text models.
            # Text-only generation remains supported through the chat template.
            if AutoModelForImageTextToText is None:
                raise causal_lm_error

            print(
                "AutoModelForCausalLM loading failed; trying "
                "AutoModelForImageTextToText instead."
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map={"": "cuda:0"},
                trust_remote_code=True,
                local_files_only=True,
                low_cpu_mem_usage=True,
                dtype=torch.bfloat16,
            )

        self.model.eval()

    def _build_prompts(self, texts: list[str]) -> list[str]:
        formatted_prompts = []

        for text in texts:
            messages = [{"role": "user", "content": str(text)}]

            if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                formatted = f"User: {text}\nAssistant:"

            formatted_prompts.append(formatted)

        return formatted_prompts

    def _chunks(self, items: list[str], chunk_size: int):
        for start in range(0, len(items), chunk_size):
            yield start, items[start:start + chunk_size]

    @torch.inference_mode()
    def invoke_batch(
        self,
        user_texts: list[str],
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> list[SimpleResponse]:
        """
        Batched generation. Returns outputs in the same order as inputs.

        temperature and max_new_tokens can be overridden per call, which is needed
        for Pipeline A:
          - guess: temperature 0.0, short output
          - description generation: temperature 0.8, longer output
        """
        if not user_texts:
            return []

        temperature = self.default_temperature if temperature is None else temperature
        max_new_tokens = self.default_max_new_tokens if max_new_tokens is None else max_new_tokens

        prompts = self._build_prompts(user_texts)
        all_responses: list[SimpleResponse] = []

        for _, prompt_chunk in self._chunks(prompts, self.max_batch_size):
            inputs = self.tokenizer(
                prompt_chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to("cuda")

            input_length = inputs.input_ids.shape[1]
            do_sample = temperature > 0

            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "repetition_penalty": self.repetition_penalty,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            if do_sample:
                generation_kwargs["temperature"] = temperature
                generation_kwargs["top_p"] = self.top_p

            out = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

            for i in range(len(prompt_chunk)):
                decoded = self.tokenizer.decode(
                    out[i][input_length:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                all_responses.append(SimpleResponse(content=decoded.strip()))

            del inputs, out
            torch.cuda.empty_cache()

        return all_responses

    def invoke(self, messages, temperature: float | None = None, max_new_tokens: int | None = None) -> SimpleResponse:
        """
        Compatibility helper for older single-call code.
        """
        m0 = messages[0]
        user_text = m0.content if hasattr(m0, "content") else str(m0)
        return self.invoke_batch(
            [user_text],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )[0]

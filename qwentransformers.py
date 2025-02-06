import logging
import time
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


class Qwen2dot5:
    def __init__(
            self,
            logger: logging.Logger,
            model_name: str
    ):
        self.logger = logger
        self.model_name = model_name
        logger.info("Starting model: %s", self.model_name)
        # default: Load the model on the available device(s)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def fix_message_for_qwen(self, messages):
        for message in messages:
            for content in message["content"]:
                if content["type"] == "image_url":
                    content["type"] = "image"
                    content["image"] = content["image_url"]["url"]
                    del content["image_url"]

    def create_openai_response(self, text, output_text):
        return {
            "id": "cmpl-03f6718535534b00b90fb88f402257c9",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": output_text[0],
                    },
                    "finish_reason": "length",
                    "index": 0,
                    "logprobs": None,
                }
            ],
            "created": int(time.time()),
            "model": self.model_name,
            "system_fingerprint": None,
            "object": "chat.completion",
            "usage": {
                "completion_tokens": len(output_text[0].split()),
                "prompt_tokens": len(text.split()),
                "total_tokens": len(output_text[0].split()) + len(text.split())
            }
        }

    def generate_kwards(self, request):
        args = {}
        if "max_completion_tokens" in request:
            args["max_new_tokens"] = request["max_completion_tokens"]
        if "temperature" in request:
            args["temperature"] = request["temperature"]
        return args

    def predict(self, request):
        self.logger.info("Request: %s", request)
        self.fix_message_for_qwen(request["messages"])

        text = self.processor.apply_chat_template(
            request["messages"], tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(request["messages"])
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        qwen_args = self.generate_kwards(request)

        generated_ids = self.model.generate(**inputs, **qwen_args)

        # Inference: Generation of the output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return self.create_openai_response(text, output_text)



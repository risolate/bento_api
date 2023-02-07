from transformers import Pipeline


class MyGenerationPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "maybe_arg" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, text, maybe_arg=2):
        input_ids = self.tokenizer(text, padding="max_length", max_length=200, return_tensors="pt")
        return input_ids

    def _forward(self, model_inputs):
        outputs = self.model.generate(
            model_inputs["input_ids"],
            max_length=model_inputs["input_ids"].size(-1) * 2,
            early_stopping=True,
            repetition_penalty=2.0,
            temperature=0.8,
            forced_bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        return outputs

    def postprocess(self, model_outputs):
        return self.tokenizer.decode(model_outputs.squeeze().detach().tolist(), skip_special_tokens=True)

import hashlib
import json
from pathlib import Path

import lotus


class MockLM(lotus.models.LM):
    def __init__(self, model_name, cache_file, mode="record"):
        super().__init__(model=model_name)
        self.cache_file = Path(cache_file)
        self.mode = mode
        self.cache = self.load_cache()

    def load_cache(self):
        if self.cache_file.exists():
            with open(self.cache_file, "r") as f:
                return json.load(f)
        return {}

    def save_cache(self):
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f)

    def hash_prompt(self, prompt):
        if not isinstance(prompt, str):
            prompt_str = json.dumps(prompt, sort_keys=True)
        else:
            prompt_str = prompt

        return hashlib.md5(prompt_str.encode("utf-8")).hexdigest()

    def __call__(self, prompts, **kwargs):
        responses = []
        new_data_recorded = False

        for p in prompts:
            p_hash = self.hash_prompt(p)

            if p_hash in self.cache:
                responses.append(self.cache[p_hash])
            elif self.mode == "record":
                lm_output = super().__call__([p], **kwargs)

                real_response = lm_output.outputs[0]
                self.cache[p_hash] = real_response
                responses.append(real_response)
                new_data_recorded = True
            else:
                responses.append("MOCK_REPLAY_MISSING")

        if new_data_recorded:
            self.save_cache()

        from lotus.models.lm import LMOutput

        return LMOutput(outputs=responses)

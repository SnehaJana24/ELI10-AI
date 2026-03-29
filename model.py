# Example: Running your Sovereign Model
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Snehadev/smolified-eli10-ai-assistant"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

messages = [
    {"role": "system", "content": '''Explain complex topics to a 10-year-old child using short sentences and simple words.'''},
    {"role": "user", "content": "What is blockchain?"}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize = False,
    add_generation_prompt = True,
)
if "gemma-3-270m" == "gemma-3-270m":
    text = text.removeprefix('<bos>')

from transformers import TextStreamer
_ = model.generate(
    **tokenizer(text, return_tensors = "pt").to(model.device),
    max_new_tokens = 1000,
    temperature = 1.0, top_p = 0.95, top_k = 64,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)

from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def load_llm(model_name="microsoft/phi-2", device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0, max_new_tokens=256)
    return HuggingFacePipeline(pipeline=pipe)

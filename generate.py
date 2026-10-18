import sys, json, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

job = json.load(open('job.json'))
prompt = job.get('prompt', '')

print("🧠 جاري تحميل الشبكة العصبية (PyTorch)...")
# تحميل نموذج شبكة عصبية مفتوح المصدر (متوافق مع PyTorch)
model_id = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu")

messages = [
    {"role": "system", "content": "أنت مبرمج خبير. اكتب الكود المطلوب فقط باللغة العربية."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt")

print("⚙️ جاري التوليد...")
outputs = model.generate(**inputs, max_new_tokens=500, temperature=0.7, do_sample=True)
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

os.makedirs('out', exist_ok=True)
json.dump({"job_id": job['job_id'], "answer": response, "model": "PyTorch-Qwen"}, open(f"out/{job['job_id']}.json", "w"), ensure_ascii=False, indent=2)

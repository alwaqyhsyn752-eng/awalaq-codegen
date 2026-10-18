import json, os
from llama_cpp import Llama

job = json.load(open('job.json'))
prompt = job.get('prompt', '')

llm = Llama(model_path="models/qwen.gguf", n_ctx=2048, n_threads=4, verbose=False)

sys_prompt = "أنت مبرمج خبير. اكتب الكود المطلوب فقط، مع شرح بسيط باللغة العربية. استخدم ``` للكود."
res = llm.create_chat_completion(
    messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}],
    max_tokens=800, temperature=0.7
)

answer = res['choices'][0]['message']['content']

os.makedirs('out', exist_ok=True)
json.dump({"job_id": job['job_id'], "answer": answer, "model": "Qwen-1.5B-GGUF"}, open(f"out/{job['job_id']}.json", "w"), ensure_ascii=False, indent=2)

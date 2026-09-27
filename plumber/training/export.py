"""plumber export: merge a trained adapter into the trunk (bf16, CPU), remove the LM head, write sharded safetensors + head.pt, optionally push.
  HF_TOKEN=... plumber export --ckpt runs/v1/step2205 --out ./plumb_merged --card ./README.md [--repo org/name]
"""
import argparse, os, shutil, time, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi
BASE = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True); ap.add_argument("--repo", default=None); ap.add_argument("--card", default="README.md")
    a = ap.parse_args(); t0 = time.time()
    lm = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True)
    print(f"[load] {time.time()-t0:.0f}s", flush=True)
    trunk = PeftModel.from_pretrained(lm.model, a.ckpt).merge_and_unload()          # LoRA folded into in_proj / q,k,v,o / shared-expert up,down
    lm.model = trunk; del lm.lm_head                                              # LM head deleted: 2688 x 131072 params gone
    lm.config.architectures = ["NemotronHModel"]; lm.config.plumb = json.load(open(os.path.join(os.path.dirname(a.card), "plumb_config.json")))
    os.makedirs(a.out, exist_ok=True)
    trunk.save_pretrained(a.out, safe_serialization=True, max_shard_size="5GB")   # saves the bare trunk (NemotronHModel) weights + config
    AutoTokenizer.from_pretrained(BASE).save_pretrained(a.out)
    shutil.copy(os.path.join(a.ckpt, "head.pt"), os.path.join(a.out, "head.pt"))
    for f in ("plumb_config.json", "README.md"): shutil.copy(os.path.join(os.path.dirname(a.card), f), os.path.join(a.out, f))
    print(f"[saved] {a.out}  {sum(os.path.getsize(os.path.join(a.out,f)) for f in os.listdir(a.out))/1e9:.1f} GB  {time.time()-t0:.0f}s", flush=True)
    if a.repo:
        api = HfApi(token=os.environ["HF_TOKEN"]); api.create_repo(a.repo, repo_type="model", exist_ok=True)
        api.upload_folder(repo_id=a.repo, folder_path=a.out, repo_type="model", commit_message="merged bf16 trunk (LM head removed) + head.pt")
        print(f"[uploaded] https://huggingface.co/{a.repo}  {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()

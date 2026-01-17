PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=3 python3 src/llms_know_difficulty/main.py --probe linear_eoi_probe --dataset livecodebench_code_generation_lite --model Qwen/Qwen2.5-Coder-32B-Instruct --max_len 4096 --k 1 --temperature 0.2

# Available probes:
# linear_eoi_probe
# tfidf_probe
# attn_probe ==> (PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)

# 0.0 linear_eoi_probe=>0.840313088740072 , tfidf=> 0.7598692816196261, benchmark SR => 0.7234, mae => 0.25115111470222473

# 0.2 linear_eoi_probe=> xx, tfidf=> 0.5004510872572084 , benchmark SR => xx, mae =>  xx

# 0.7 linear_eoi_probe=> 0.6896151880607202, tfidf=> 0.5300055575966742, benchmark SR => 0.7272, mae => 0.20852836966514587

# 1.0 linear_eoi_probe=> 0.7417859232384286 tfidf=> 0.5788201324299777, benchmark SR => 0.6887, mae =>0.19418743252754211




# ### maj voting (t=0.7, benchmark maj@50=0.7798, sr=0.727188, pass@50: 0.9032)
# tfidf probe => 0.7669958458339186, linear_eoi_probe =>0.8461524753430105
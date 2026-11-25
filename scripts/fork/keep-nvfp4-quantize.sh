# 1. convert: NVFP4 groups repacked as-is, everything else BF16 (not --fp8-as-q8, that would requantize later)
python3 convert_hf_to_gguf.py <checkpoint-dir> --outfile <nvfp4-bf16.gguf> --outtype bf16

# 2. pin every NVFP4 tensor to nvfp4 (pinned = copied unchanged)
python3 gguf-py/gguf/scripts/gguf_dump.py <nvfp4-bf16.gguf> \
  | awk -F'|' '$3 ~ /NVFP4/ {gsub(/ /,"",$4); print "^" $4 "$=nvfp4"}' > nvfp4-keep.txt
wc -l nvfp4-keep.txt

# 3. dry run: shows each tensor's new type and the total size, writes nothing
build/bin/llama-quantize --dry-run --tensor-type-file nvfp4-keep.txt <nvfp4-bf16.gguf> <hybrid.gguf> Q4_K_M

# 4. real run (no --allow-requantize: if it complains, an NVFP4 tensor slipped through), then the d2t copy
build/bin/llama-quantize --tensor-type-file nvfp4-keep.txt <nvfp4-bf16.gguf> <hybrid.gguf> Q4_K_M
python3 scripts/fork/mtp-d2t.py <hybrid.gguf> <hybrid-d2t.gguf> --n-draft 49152 --table <table.bin>

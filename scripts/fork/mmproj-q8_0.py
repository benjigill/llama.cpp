#!/usr/bin/env python3
# Requantize an F16 mmproj GGUF to Q8_0 (llama-quantize does not handle mmproj files).
# Same rules as convert_hf_to_gguf.py --mmproj --outtype q8_0: 2D weights become Q8_0,
# patch/position embeddings, norms and biases keep their type.
#
# usage: python3 scripts/fork/mmproj-q8_0.py <mmproj-f16.gguf> <mmproj-q8_0.gguf>

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))

import gguf  # noqa: E402

KEEP = (".patch_embd.", ".patch_merger.", ".position_embd.")


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <in.gguf> <out.gguf>")

    reader = gguf.GGUFReader(sys.argv[1])
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    writer = gguf.GGUFWriter(sys.argv[2], arch=arch, endianess=reader.endianess)

    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        value = field.contents()
        if field.name == gguf.Keys.General.FILE_TYPE:
            value = gguf.LlamaFileType.MOSTLY_Q8_0
        writer.add_key_value(field.name, value, val_type, sub_type=sub_type)

    n_q = 0
    tensors = []
    for t in reader.tensors:
        data, qtype = t.data, t.tensor_type
        quantize = (
            qtype in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16, gguf.GGMLQuantizationType.F32)
            and len(data.shape) == 2
            and t.name.endswith(".weight")
            and not any(k in t.name for k in KEEP)
            and data.shape[-1] % 32 == 0
        )
        if quantize:
            f32 = gguf.quants.dequantize(data, qtype)
            data, qtype = gguf.quants.quantize(f32, gguf.GGMLQuantizationType.Q8_0), gguf.GGMLQuantizationType.Q8_0
            n_q += 1
        tensors.append((t.name, data, qtype))
        writer.add_tensor_info(t.name, data.shape, data.dtype, data.nbytes, qtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for _, data, _ in tensors:
        writer.write_tensor_data(data, tensor_endianess=reader.endianess)
    writer.close()

    print(f"quantized {n_q}/{len(tensors)} tensors to Q8_0 -> {sys.argv[2]}")


if __name__ == "__main__":
    main()

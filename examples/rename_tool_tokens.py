import shutil
from transformers import AutoTokenizer
from omnitok.io import rename_reserved_token

input_dir  = "/capstor/scratch/cscs/dtamayomela/tokenizer_fixed"
output_dir = "/capstor/scratch/cscs/dtamayomela/tokenizer_with_tool"

shutil.copytree(input_dir, output_dir)

tok = AutoTokenizer.from_pretrained(output_dir)
rename_reserved_token(output_dir, tok, tok.convert_ids_to_tokens(73), "<|tool_output_start|>")
rename_reserved_token(output_dir, tok, tok.convert_ids_to_tokens(74), "<|tool_output_end|>")
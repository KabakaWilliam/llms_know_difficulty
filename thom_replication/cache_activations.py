# Load dataset
# Load model

# For each prompt
# Tokenize 
# Put through model
# Stack activations into N_Points X N_Layers X Hidden_Size tensor
# Store activations to disk, data/activations/{dataset}/{split}/{model}/layer_{num}
# or as data/activations/{dataset}/{model}/layer_{num}

from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from tqdm import tqdm
import numpy as np
import zstandard as zstd
import io


def cache_activations(
        model_name, 
        dataset_name, 
        max_questions_per_split, 
        output_dir,
        prompt_suffix,
    ):

    ds = load_dataset(dataset_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    model.eval()
    model.to('cuda')

    for split in ds:

        question_activations = [] # Shape: (N_Points, N_Layers, Seq_Len, Hidden_Size)

        for idx, item in tqdm(enumerate(ds[split]), desc=f"Generating activations for split {split}"):

            if max_questions_per_split is not None and idx >= max_questions_per_split:
                break

            # important this is the same as in "create_success_rate_datasets.py"
            prompt = item['problem'] + ' ' + prompt_suffix
            messages = [
                {"role": "user", "content": prompt}
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states  # Tuple of (layer_count, batch_size, seq_len, hidden_size)

                # Stack hidden states into a single tensor
                hidden_states = torch.stack(hidden_states, dim=1)  # Shape: (batch_size, N_Layers, Seq_Len, Hidden_Size)
                hidden_states = hidden_states.squeeze(0).cpu()  # Remove batch dimension, Shape: (N_Layers, Seq_Len, Hidden_Size)

                question_activations.append(hidden_states)


        # Save activations per layer per question
        for idx, activations in tqdm(enumerate(question_activations), desc="Saving activations"):
            n_layers = activations.shape[0]
            for layer_idx in range(n_layers):
                save_dir = os.path.join(output_dir, dataset_name.replace('/', '_'), split, f"question_{idx}", model_name.replace('/', '_'))
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'layer_{layer_idx}.npz')
                layer_activations = activations[layer_idx, :, :]  # Shape: (N_Points, Seq_Len, Hidden_Size)
                np.savez_compressed(save_path, data=layer_activations.cpu().numpy())
                # torch.save(layer_activations, save_path)


        # below is CODE I USED TO BENCHMARK DIFFERENT COMPRESSION ALGORITHMS 

        # # activations are list of length N_Points, each element is (N_Layers, Seq_Len, Hidden_Size)
        # # Seq_Len may vary per question
        # # First find the longest Seq_Len
        # max_seq_len = max(act.shape[1] for act in question_activations)
        # # Pad all activations to max_seq_len
        # for i in range(len(question_activations)):
        #     act = question_activations[i]
        #     seq_len = act.shape[1]
        #     if seq_len < max_seq_len:
        #         pad_size = max_seq_len - seq_len
        #         pad_tensor = torch.zeros((act.shape[0], pad_size, act.shape[2]))
        #         question_activations[i] = torch.cat([act, pad_tensor], dim=1)  # Pad on the right
        # # Now stack into a single tensor
        # question_activations = torch.stack(question_activations, dim=0)  # Shape: (N_Points, N_Layers, Seq_Len, Hidden_Size)

        # # save activation per layer for all questions
        # n_layers = question_activations.shape[1]
        # for layer_idx in range(n_layers):
        #     save_dir = os.path.join(output_dir, dataset_name.replace('/', '_'), split, model_name.replace('/', '_'))
        #     os.makedirs(save_dir, exist_ok=True)
        #     layer_activations = question_activations[:, layer_idx, :, :]  # Shape: (N_Points, Seq_Len, Hidden_Size)

        #     # save_path = os.path.join(save_dir, f'layer_{layer_idx}.npz')
        #     # np.savez_compressed(save_path, data=layer_activations.cpu().numpy())
        #     # torch.save(layer_activations, save_path)

        #     buf = io.BytesIO()
        #     torch.save(layer_activations, buf)

        #     compressed = zstd.ZstdCompressor(level=10).compress(buf.getvalue())

        #     save_path = os.path.join(save_dir, f'layer_{layer_idx}.zst')
        #     with open(save_path, "wb") as f:
        #         f.write(compressed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use for generating activations')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset to load')
    parser.add_argument('--max_questions_per_split', type=int, default=1000, help='Maximum number of questions to process per split')
    parser.add_argument('--output_dir', type=str, default="data/activations", help='Directory to save the activations')
    parser.add_argument('--prompt_suffix', type=str, default="Let's think step by step and output the final answer within \\boxed{}.", help='Suffix to append to each prompt')
    args = parser.parse_args()

    cache_activations(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        max_questions_per_split=args.max_questions_per_split,
        output_dir=args.output_dir,
        prompt_suffix=args.prompt_suffix,
    )
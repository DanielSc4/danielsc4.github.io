import torch
import numpy as np
from transformers import AutoModelForCausalLM
from tqdm import tqdm

from .utils.model_utils import rsetattr, rgetattr
from transformers import AutoTokenizer

def split_activation(activations, config):
    """split the residual stream (d_model) into n_heads activations for each layer

    Args:
        activations (list[torch.Tensor]): list of residual streams for each layer shape: (n_layers, batch, seq, d_model)
        config (dict[str, Any]): model's config

    Returns:
        activations: reshaped activation in [n_layers, seq, n_heads, d_head]
    """
    new_shape = torch.Size([
        activations[0].shape[0],         # batch_size == 1
        activations[0].shape[1],         # seq_len
        config['n_heads'],                          # n_heads
        config['d_model'] // config['n_heads'],     # d_head
    ])
    attn_activations = torch.vstack([act.view(*new_shape) for act in activations])
    return attn_activations


def extract_activations(
        tokenized_prompts: list[torch.Tensor], 
        model: AutoModelForCausalLM, 
        config: dict[str, any],
        tokenizer: AutoTokenizer,
        device: str,
    ):
    """Extract the activation and the output produced from the model using the tokenized prompts provided

    Args:
        tokenized_prompts (list[torch.Tensor]): list of tokenized prompts
        model (AutoModelForCausalLM): HuggingFace model
        config (dict[str, any]): model's config
        tokenizer (AutoTokenizer): HuggingFace tokenizer
        device (str): device

    Returns:
        tuple[list[torch.Tensor], list[torch.Tensor]]: tuple corresponding to the activations and the model output
    """
    dataset_activations, outputs = [], []
    for prompt in tqdm(tokenized_prompts, total = len(tokenized_prompts), desc = '[x] Extracting activations'):
        prompt = prompt.to(device)
        with model.generate(max_new_tokens=1, pad_token_id=tokenizer.pad_token_id) as generator:
            # invoke works in a generation context, where operations on inputs and outputs are tracked
            with generator.invoke(prompt) as invoker:
                layer_attn_activations = []
                for layer_name in config['attn_hook_names']:
                    layer_attn_activations.append(rgetattr(model, layer_name).output.save())
        outputs.append(generator.output)

        # get the values from the activations
        layer_attn_activations = [att.value for att in layer_attn_activations]
        
        # from hidden state split heads and permute: n_layers, tokens, n_heads, d_head -> n_layers, n_heads, tokens, d_head
        attn_activations = split_activation(layer_attn_activations, config).permute(0, 2, 1, 3)
        dataset_activations.append(attn_activations)
    return dataset_activations, outputs


def get_mean_activations(
        tokenized_prompts: list[torch.Tensor], 
        important_ids: list[int], 
        tokenizer: AutoTokenizer,
        model: AutoModelForCausalLM, 
        config: dict[str, any],
        correct_labels: list[str],
        device: str,
    ):
    """Compute the average of all the model's activation on the provided prompts

    Args:
        tokenized_prompts (list[torch.Tensor]): list of tokenized prompts
        important_ids (list[int]): list of important indexes i.e. the tokens where the average is computed
        tokenizer (AutoTokenizer): HuggingFace tokenizer
        model (AutoModelForCausalLM): HuggingFace model
        config (dict[str, any]): model's config
        correct_labels (list[str]): list of correct labels for each ICL prompt
        device (str): device

    Returns:
        torch.Tensor: mean of activations (`n_layers, n_heads, seq_len, d_head`)
    """

    activations, outputs = extract_activations(
        tokenized_prompts=tokenized_prompts, 
        model=model, 
        config=config,
        tokenizer=tokenizer,
        device=device,
    )

    # keep only important tokens
    activations_clean = torch.stack(
        [activations[i][:, :, important_ids[i], :] for i in range(len(activations))]
    )

    # considering only the first token to evaluate the output
    only_output_tokens = np.array(list(map(lambda x: x.squeeze()[-1].item(), outputs)))
    only_labels_tokens = np.array([ele[0] for ele in tokenizer(correct_labels)['input_ids']])

    correct_idx = (only_output_tokens == only_labels_tokens)
    accuracy = correct_idx.sum() / len(correct_idx)
    if correct_idx.sum() > 0:
        print(f'[x] Model accuracy: {accuracy:.2f}, using {correct_idx.sum()} (out of {len(correct_idx)}) examples to compute mean activations')
    else:
        print(f'[x] Model accuracy is 0, mean_activations cannot be computed!')
        return None

    # using only activations from correct prediction to compute the mean_activations
    correct_activations = activations_clean[correct_idx]
    
    mean_activations = correct_activations.mean(axis = 0)
    
    
    return mean_activations
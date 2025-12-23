import argparse

import torch
import torch.nn.functional as F

def activation(n, num_layers, over_zero, languages, mask_file_path):
    top_rate = 0.01
    filter_rate = 0.95
    activation_bar_ratio = 0.95

    # ------------------------------------------------------------------
    # 1. Activation probability p_{l,h,k} = over_zero / n_k
    # ------------------------------------------------------------------
    activation_probs = over_zero / n # layer x inter x lang_num # [layer, neuron, lang]

    # Normalize across languages for entropy
    normed_activation_probs = activation_probs / activation_probs.sum(dim=-1, keepdim=True)
    normed_activation_probs[torch.isnan(normed_activation_probs)] = 0

    log_probs = torch.where(normed_activation_probs > 0, normed_activation_probs.log(), 0)

    entropy = -torch.sum(normed_activation_probs * log_probs, dim=-1) # [layer, neuron]
    largest = False
    
    if torch.isnan(entropy).sum():
        print(torch.isnan(entropy).sum())
        raise ValueError
    
    if torch.isnan(entropy).any():
        raise ValueError("NaN detected in entropy — check activation statistics.")
    
    # ------------------------------------------------------------------
    # 2. Filter out low-activity neurons
    # ------------------------------------------------------------------
    flattened_probs = activation_probs.flatten()
    top_prob_value = flattened_probs.kthvalue(round(len(flattened_probs) * filter_rate)).values.item()
    
    # print(top_prob_value)
    print(f"\n[INFO] High-activity cutoff (top {int((1-filter_rate)*100)}% firing probability):")
    print(f"       activation_prob >= {top_prob_value:.6f}")
    
    # dismiss the neruon if no language has an activation value over top 90%
    top_position = (activation_probs > top_prob_value).sum(dim=-1)
    entropy[top_position == 0] = -torch.inf if largest else torch.inf

    # ------------------------------------------------------------------
    # 3. Select lowest-entropy neurons (LAPE)
    # ------------------------------------------------------------------
    flattened_entropy = entropy.flatten()
    top_entropy_value = round(len(flattened_entropy) * top_rate)

    _, index = flattened_entropy.topk(top_entropy_value, largest=largest)
    row_index = index // entropy.size(1)
    col_index = index % entropy.size(1)

    selected_probs = activation_probs[row_index, col_index] # n x lang
    # for r, c in zip(row_index, col_index):
    #     print(r, c, activation_probs[r][c])
    print(f"\n[INFO] LAPE-selected neurons:")
    print(f"       Selected {selected_probs.size(0)} neurons "
          f"(top {top_rate*100:.1f}% lowest entropy)")
    
    # ------------------------------------------------------------------
    # 4. Dominant language per neuron
    # ------------------------------------------------------------------
    dominant_lang = selected_probs.argmax(dim=-1)
    lang_counts = torch.bincount(dominant_lang, minlength=len(languages))

    print("\n[INFO] Dominant language per selected neuron:")
    for lang, count in zip(languages, lang_counts.tolist()):
        print(f"       {lang:>3}: {count} neurons")

    # print(selected_probs.size(0), torch.bincount(selected_probs.argmax(dim=-1)))
    selected_probs = selected_probs.transpose(0, 1)
    activation_bar = flattened_probs.kthvalue(round(len(flattened_probs) * activation_bar_ratio)).values.item()
    # print((selected_probs > activation_bar).sum(dim=1).tolist())
    strong_mask = selected_probs > activation_bar
    strong_counts = strong_mask.sum(dim=1)

    print(f"\n[INFO] Strong language-specific neurons "
          f"(activation_prob >= {activation_bar:.6f}):")

    for lang, count in zip(languages, strong_counts.tolist()):
        print(f"       {lang:>3}: {count} neurons")

    # ------------------------------------------------------------------
    # 6. Build per-language per-layer mask
    # ------------------------------------------------------------------
    lang, indice = torch.where(selected_probs > activation_bar)

    merged_index = torch.stack((row_index, col_index), dim=-1)
    final_indice = []
    for _, index in enumerate(indice.split(torch.bincount(lang).tolist())):
        lang_index = [tuple(row.tolist()) for row in merged_index[index]]
        lang_index.sort()
        layer_index = [[] for _ in range(num_layers)]
        for l, h in lang_index:
            layer_index[l].append(h)
        for l, h in enumerate(layer_index):
            layer_index[l] = torch.tensor(h).long()
        final_indice.append(layer_index)
    torch.save(final_indice, mask_file_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", type=str, nargs='+', default=['id', 'vi', 'zh'])
    parser.add_argument("--data_prefix", type=str, default="data/parallel-only-7B-34B/ckpt-34")
    parser.add_argument("--model_type", type=str, default="olmo2")
    parser.add_argument("--mask_file_path", type=str, default="activation_mask/olmo2")
    args = parser.parse_args()

    languages = args.languages

    n, over_zero = [], []
    for language in languages:
        data = torch.load(f'{args.data_prefix}/activation.{language}.train.{args.model_type}')
        n.append(data['n'])
        over_zero.append(data['over_zero'])

    n = torch.tensor(n)
    over_zero = torch.stack(over_zero, dim=-1)

    num_layers, intermediate_size, lang_num = over_zero.size()

    print("\n[INFO] Loaded activation statistics")
    print(f"       Languages        : {languages}")
    print(f"       Num layers       : {num_layers}")
    print(f"       Intermediate size: {intermediate_size}")
    print(f"       Total neurons    : {num_layers * intermediate_size}")

    activation(n=n, 
               num_layers=num_layers, 
               over_zero=over_zero, 
               languages=languages,
               mask_file_path=args.mask_file_path)

if __name__ == "__main__":
    main()

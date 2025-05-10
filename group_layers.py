import yaml
import argparse
import numpy as np
from collections import defaultdict
from sklearn.cluster import SpectralClustering

import torch

def main(args):
    similarity = torch.load(args.cka_similarity)
    affinity = similarity.cpu().numpy()
    if args.weighted_by_index:
        # Weight by inverse index distance
        index_matrix = np.abs(np.subtract.outer(np.arange(32), np.arange(32)))
        scale = 20.0
        index_weight = np.exp(-index_matrix / scale)  # shape (32, 32)
        # Final affinity matrix: similarity × index_weight
        affinity = affinity * index_weight
        # Ensure symmetry (optional but good practice)
        affinity = (affinity + affinity.T) / 2
    np.fill_diagonal(affinity, 1)
    clustering = SpectralClustering(n_clusters=8, affinity='precomputed', random_state=0)
    labels = clustering.fit_predict(affinity)
    print("Cluster labels:", labels)

    # Dummy config dictionary (parsed from YAML)
    xKV_config = {
        'num_layers': 32,
        'layer_merge_impl': 'svd',
        'rank_k': 512,
        'rank_v': 768,
        'slerp_t': 0.5,
        'slerp_gamma': 0.05,
        'merge_key': True,
        'merge_value': True,
        'layer_groups': []  # will be replaced
    }

    print("Cluster labels:", labels)
    # Group layers by cluster label
    group_dict = defaultdict(list)
    group_idx = 0
    for layer, label in enumerate(labels):
        if layer > 0 and labels[layer] != labels[layer - 1]:
            # Move to the next group
            group_idx += 1
        # Continue adding to the current group
        group_dict[group_idx].append(layer)
    # Sort groups by label index for readability
    sorted_groups = sorted(group_dict.items())

    # Create new layer_groups entry
    new_layer_groups = []
    for _, layers in sorted_groups:
        group_entry = {
            'layers': sorted(layers),
            'rank_k': xKV_config['rank_k'],
            'rank_v': xKV_config['rank_v']
        }
        new_layer_groups.append(group_entry)

    # Replace old groups with new ones
    xKV_config['layer_groups'] = new_layer_groups

    # Dump updated config to YAML
    yaml_str = yaml.dump({'xKV_config': xKV_config}, sort_keys=False)
    print(yaml_str)

    with open(args.output_config, "w") as f:
        f.write(yaml_str)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Cluster layers using SpectralClustering based on similarity matrix.")

    parser.add_argument(
        "--cka_similarity",
        type=str,
        required=True,
        help="Path to the .pt file containing the (32, 32) similarity matrix (torch format)."
    )

    parser.add_argument(
        "--output_config",
        type=str,
        default="configs/grouped_layers.yaml",
        help="Output YAML file to save the generated xKV layer group config."
    )

    parser.add_argument(
        "--weighted_by_index",
        action="store_true",
        help="If set, weight similarity by index distance before clustering."
    )

    args = parser.parse_args()
    main(args)
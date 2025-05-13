import yaml
import argparse
import numpy as np
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering

import torch

def main(args):
    similarity = torch.load(args.cka_similarity)
    similarity = similarity.cpu().numpy()
    np.fill_diagonal(similarity, 1)
    if args.weighted_by_index:
        if args.verbose:
            print("Weighting similarity by index distance, scale:", args.index_scale)
        # Weight by inverse index distance
        index_matrix = np.abs(np.subtract.outer(np.arange(32), np.arange(32)))
        scale = args.index_scale
        index_weight = np.exp(-index_matrix / scale)  # shape (32, 32)
        # Final affinity matrix: similarity × index_weight
        similarity = similarity * index_weight
        # Ensure symmetry (optional but good practice)
        similarity = (similarity + similarity.T) / 2
    if args.verbose:
        print("Number of groups:", args.ngroups)
    dissimilarity = 1 - similarity
    clustering = AgglomerativeClustering(
        n_clusters=args.ngroups,
        metric='precomputed',
        linkage='average',
        # connectivity=connectivity
    )
    labels = clustering.fit_predict(dissimilarity)
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
    if args.verbose:
        print("Cluster labels:", labels)
    # Group layers by cluster label
    group_dict = defaultdict(list)
    group_idx = 0
    for layer, label in enumerate(labels):
        # we only take adjecent labels as a group
        if layer > 0 and labels[layer] != labels[layer - 1]:
            group_idx += 1
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
    if args.verbose:
        print("Generated YAML config:")
        print(yaml_str)
    
    print("Saving to:", args.output_config)
    with open(args.output_config, "w") as f:
        f.write(yaml_str)

if __name__ == "__main__":
    """
    Usage:
    python group_layers.py --cka_similarity l31-8b-ruler-mv-cka.pt --verbose --weighted_by_index
    """
    parser = argparse.ArgumentParser(description="Cluster layers using SpectralClustering based on similarity matrix.")

    parser.add_argument(
        "--cka_similarity",
        type=str,
        required=True,
        help="Path to the .pt file containing the (#layer, #layer) similarity matrix (torch format)."
    )
    parser.add_argument(
        "--ngroups",
        type=int,
        default=8,
        help="Number of groups to cluster the layers into."
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
    parser.add_argument(
        "--index_scale",
        type=float,
        default=50.0,
        help="The scale to use for the index distance weighting."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, print additional information."
    )
    args = parser.parse_args()
    main(args)
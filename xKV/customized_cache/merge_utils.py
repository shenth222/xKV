import yaml

def generate_merge_groups(method="consecutive", group_size=None, start=None, end=None, file_path=None):
    """
    Generate merge_groups based on the specified method.

    Args:
        method (str): "consecutive" or "customized".
        group_size (int, optional): Number of layers per group (for "consecutive" mode).
        start (int, optional): Start layer index (for "consecutive" mode).
        end (int, optional): End layer index (for "consecutive" mode).
        file_path (str, optional): Path to file containing layer groupings (for "customized" mode).

    Returns:
        list of tuples: List of (start_layer, end_layer) pairs.
    """
    if method == "consecutive":
        # Validate inputs
        if group_size is None or start is None or end is None:
            raise ValueError("For 'consecutive' mode, group_size, start, and end must be provided.")

        merge_groups = [
            (i, min(i + group_size - 1, end)) for i in range(start, end + 1, group_size)
        ]

    elif method == "customized":
        if file_path is None:
            raise ValueError("For 'customized' mode, file_path must be provided.")
        
        try:
            with open(file_path, "r") as f:
                data = yaml.safe_load(f)
                merge_groups = [tuple(group) for group in data.get("merge_groups", [])]
        except Exception as e:
            raise ValueError(f"Error reading YAML file {file_path}: {e}")

    else:
        raise ValueError("Invalid method. Choose 'consecutive' or 'customized'.")

    return merge_groups

import os
import torch
import matplotlib.pyplot as plt
import argparse
from typing import Tuple, List, Dict
from sae_lens.activation_visualization import (
    load_llava_model, load_sae, prepare_input, generate_with_saev,
    map_patches_to_image, overlay_activation_on_image
)

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Activation Visualization Script")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the Llava model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Llava model checkpoint.")
    parser.add_argument("--device", type=str, required=True, help="Device for Llava model.")
    parser.add_argument("--sae_device", type=str, required=True, help="Device for SAE model.")
    parser.add_argument("--sae_path", type=str, required=True, help="Path to the SAE model checkpoint.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save visualizations.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for the model.")
    return parser.parse_args()

def initialize_models(
    model_name: str, model_path: str, device: str, sae_path: str, sae_device: str, n_devices: int
) -> Tuple:
    """
    Load the required models including Llava and SAE.

    Args:
        model_name (str): Name of the Llava model.
        model_path (str): Path to the Llava model checkpoint.
        device (str): Device for Llava model.
        sae_path (str): Path to the SAE model checkpoint.
        sae_device (str): Device for SAE model.
        n_devices (int): Number of devices for multi-GPU setup.

    Returns:
        Tuple: Processor, vision model, multi-modal projector, hooked language model, SAE.
    """
    processor, vision_model, _, multi_modal_projector, hook_language_model = load_llava_model(
        model_name, model_path, device, n_devices=n_devices
    )
    sae = load_sae(sae_path, sae_device)
    return processor, vision_model, multi_modal_projector, hook_language_model, sae

def process_input(
    processor, device: str, image_path: str, prompt: str
) -> Tuple[Dict, torch.Tensor]:
    """
    Prepare input for the model.

    Args:
        processor: The processor for input preparation.
        device (str): Device for processing.
        image_path (str): Path to the input image.
        prompt (str): Text prompt for the model.

    Returns:
        Tuple[Dict, torch.Tensor]: Processed inputs and image tensor.
    """
    return prepare_input(processor, device, image_path, prompt)

def visualize_activations(
    inputs: Dict, image: torch.Tensor, save_path: str, hook_language_model, sae, sae_device: str
) -> None:
    """
    Generate and visualize activations.

    Args:
        inputs (Dict): Processed model inputs.
        image (torch.Tensor): Input image tensor.
        save_path (str): Path to save visualizations.
        hook_language_model: Hooked language model.
        sae: SAE model.
        sae_device (str): Device for SAE model.
    """
    total_activation_l1_norms_list, patch_features_list, feature_act_list, image_indices = generate_with_saev(
        inputs, hook_language_model, processor, save_path, image, sae, sae_device
    )

    # Extract activation map for visualization
    current_activation_map = map_patches_to_image(total_activation_l1_norms_list[0], max_val=1000)
    final_image = overlay_activation_on_image(image, current_activation_map, alpha=128)
    final_image.save(os.path.join(save_path, "activation_visualization.png"))

def analyze_activation_statistics(
    total_activation_l1_norms_list: List[torch.Tensor]
) -> None:
    """
    Analyze and plot activation L1 norms.

    Args:
        total_activation_l1_norms_list (List[torch.Tensor]): List of activation L1 norms.
    """
    global_activation_l1_norms = torch.cat(total_activation_l1_norms_list).flatten()
    global_activation_l1_norms_np = global_activation_l1_norms.cpu().numpy()

    # Shift values if necessary
    min_value = global_activation_l1_norms_np.min()
    if min_value <= 0:
        global_activation_l1_norms_np += abs(min_value) + 1e-6

    low_values = global_activation_l1_norms_np[global_activation_l1_norms_np < 1500]
    high_values = global_activation_l1_norms_np[global_activation_l1_norms_np >= 1500]

    # Plot histogram for low values
    plt.figure(figsize=(10, 6))
    plt.hist(low_values, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Low Range of Activation L1 Norms', fontsize=16)
    plt.xlabel('Activation L1 Norms', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.show()
    print("High values:", high_values)

if __name__ == "__main__":
    args = parse_arguments()

    processor, vision_model, multi_modal_projector, hook_language_model, sae = initialize_models(
        args.model_name, args.model_path, args.device, args.sae_path, args.sae_device, n_devices=8
    )

    inputs, image = process_input(processor, args.device, args.image_path, args.prompt)
    visualize_activations(inputs, image, args.save_path, hook_language_model, sae, args.sae_device)

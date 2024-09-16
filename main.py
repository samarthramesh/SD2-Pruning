import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='sd-2', help='Model path of unpruned model. If not present will download and save at this path.')
    parser.add_argument('--save_model_path', type=str, default='sd-2', help='Path to save pruned model.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for selecting the MSCOCO data for evaluation')
    parser.add_argument('--text_pruning_method', type=str, default='magnitude', help="Pruning algorithm to be applied for text portion of model.")
    parser.add_argument('--image_pruning_method', type=str, default='magnitude', help="Pruning algorithm to be applied for image portion of model.")
    parser.add_argument('--text_sparsity', type=float, default=0, help="Pruning sparsity for text portion of model.")
    parser.add_argument('--image_sparsity', type=float, default=0, help="Pruning sparsity for image portion of model.")
    parser.add_argument('--metric_images', type=int, default=10000, help="Number of images to be generated for evaluation metrics.")


if __name__ == "__main__":
    main()
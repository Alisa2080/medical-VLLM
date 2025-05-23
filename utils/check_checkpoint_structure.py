import torch

def inspect_checkpoint(checkpoint_path):
    """
    加载并检查 PyTorch (Lightning) checkpoint 文件的内容。

    参数:
        checkpoint_path (str): checkpoint 文件的路径。
    """
    try:
        # 加载 checkpoint 文件
        # map_location='cpu' 确保即使是在没有 GPU 的机器上也能加载
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"Successfully loaded checkpoint from: {checkpoint_path}\n")

        print("Top-level keys in the checkpoint file:")
        for key in checkpoint.keys():
            print(f"- {key}")
        print("-" * 40)

        # 通常，模型权重存储在 'state_dict' 键下
        if 'state_dict' in checkpoint:
            print("\nKeys within 'state_dict' (model weights):")
            state_dict = checkpoint['state_dict']
            for i, (key, value) in enumerate(state_dict.items()):
                print(f"  - {key}: \t{value.shape}, \tdtype: {value.dtype}")
                if i < 20: # 打印前20个权重键以避免输出过长
                    pass
                elif i == 20:
                    print(f"  ... (and {len(state_dict) - 20} more keys)")
                    break
            print("-" * 40)
        else:
            print("\n'state_dict' key not found. The checkpoint might directly contain weights or have a different structure.")
            print("If so, inspect other top-level keys for model weights.")
            # 如果顶层直接是 state_dict (例如非Lightning保存的模型)
            is_likely_direct_state_dict = all(isinstance(v, torch.Tensor) for v in checkpoint.values())
            if is_likely_direct_state_dict and len(checkpoint.keys()) > 5: # Heuristic
                 print("\nThe checkpoint itself might be a state_dict. Printing some keys:")
                 for i, (key, value) in enumerate(checkpoint.items()):
                    print(f"  - {key}: \t{value.shape}, \tdtype: {value.dtype}")
                    if i < 20:
                        pass
                    elif i == 20:
                        print(f"  ... (and {len(checkpoint) - 20} more keys)")
                        break


        # 你还可以打印其他你感兴趣的键，例如：
        if 'epoch' in checkpoint:
            print(f"\nEpoch: {checkpoint['epoch']}")
        
        if 'global_step' in checkpoint:
            print(f"Global Step: {checkpoint['global_step']}")

        if 'hyper_parameters' in checkpoint:
            print(f"\nHyperparameters (hparams):")
            for k, v in checkpoint['hyper_parameters'].items():
                print(f"  - {k}: {v}")
            print("-" * 40)
        
        # 如果你的 BEiTLightningModule (来自 pretraining.py) 包含 optimizer_states
        if 'optimizer_states' in checkpoint and checkpoint['optimizer_states']:
            print(f"\nOptimizer states found for {len(checkpoint['optimizer_states'])} optimizer(s).")
            # print(f"Optimizer state keys for the first optimizer: {checkpoint['optimizer_states'][0].keys()}")

        # 如果你的 BEiTLightningModule 包含 lr_schedulers
        if 'lr_schedulers' in checkpoint and checkpoint['lr_schedulers']:
            print(f"\nLR scheduler states found for {len(checkpoint['lr_schedulers'])} scheduler(s).")
            # print(f"LR scheduler state for the first scheduler: {checkpoint['lr_schedulers'][0]}")


    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
    except Exception as e:
        print(f"An error occurred while loading or inspecting the checkpoint: {e}")

if __name__ == '__main__':
    # 将下面的路径替换为你的 last.ckpt 文件的实际路径
    ckpt_path = r"E:\article_code\output\beit2\finetuning_pl\mil_checkpoints\version_0\last.ckpt" 
    # 或者如果你在Linux/macOS上：
    # ckpt_path = "/path/to/your/last.ckpt"
    
    inspect_checkpoint(ckpt_path)
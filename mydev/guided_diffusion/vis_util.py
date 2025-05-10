import plotly
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import wandb

def plot_distance(pred_batch, gt_batch, self_obj, logger, tag="distance_error", step=None):
    """
    pred_batch, gt_batch: numpy arrays of shape (B, N, 3)
    logger: a W&B logger with a .log() method (e.g., wandb)
    """

    assert pred_batch.shape == gt_batch.shape, "Shape mismatch"
    B = pred_batch.shape[0]

    # Compute axis-wise squared error
    error = pred_batch - gt_batch                   # (B, N, 3)
    sq_error = error ** 2                           # (B, N, 3)
    mse_axis = sq_error.mean(axis=1)                # (B, 3): per-sample MSE for x,y,z
    rmse_axis = np.sqrt(mse_axis)                   # (B, 3): per-sample RMSE for x,y,z

    # Compute 3D distance error per sample
    distance_error = np.linalg.norm(error, axis=-1).mean(axis=1)  # (B,)

    # Plot
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].bar(range(B), rmse_axis[:, 0], color='skyblue')
    axs[0].set_title("X-axis RMSE")
    axs[0].set_xlabel("Sample")
    axs[0].set_ylabel("Error")

    axs[1].bar(range(B), rmse_axis[:, 1], color='salmon')
    axs[1].set_title("Y-axis RMSE")
    axs[1].set_xlabel("Sample")

    axs[2].bar(range(B), rmse_axis[:, 2], color='lightgreen')
    axs[2].set_title("Z-axis RMSE")
    axs[2].set_xlabel("Sample")

    axs[3].bar(range(B), distance_error, color='orchid')
    axs[3].set_title("3D Distance Error")
    axs[3].set_xlabel("Sample")

    plt.tight_layout()

    # Log to W&B
    wandb.log({tag: wandb.Image(fig)} if step is None else {tag: wandb.Image(fig), "step": step})
    plt.close(fig)

    

def plot_2d(pred_batch, gt_batch, self_obj, logger):
    # View settings: (elev, azim, title)
    views = [
        (90, -90, "XY"),   # Top-down
        (0, -90, "XZ"),    # Side
        (0, 0, "YZ")       # Front
    ]
    assert gt_batch.shape == pred_batch.shape
    B = gt_batch.shape[0]
    fig = plt.figure(figsize=(12, 4 * B))

    for i in range(B):
        gt = gt_batch[i]
        pred = pred_batch[i]
        
        for j, (elev, azim, view_name) in enumerate(views):
            ax = fig.add_subplot(B, 3, i * 3 + j + 1, projection='3d')
            ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], c='red', label='GT', s=5)
            ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c='blue', label='Pred', s=5)
            ax.view_init(elev=elev, azim=azim)
            if i == 0:
                ax.set_title(f"{view_name} View")
            if j == 0:
                ax.set_ylabel(f"Sample {i}")
            if j == 2:
                ax.legend(loc='upper right')

    plt.tight_layout()
    wandb.log({"All_Projections": wandb.Image(fig)})
    plt.close(fig)

def test_plot():
    import wandb
    # Initialize W&B
    # Initialize W&B
    wandb.init(project="your_project_name")

    # Dummy data: (B, N, 3)
    B = 4   # Number of samples in batch
    N = 100

    gt_batch = np.random.rand(B, N, 3)
    pred_batch = gt_batch + np.random.normal(scale=0.05, size=(B, N, 3))

    # View settings: (elev, azim, title)
    views = [
        (90, -90, "XY"),   # Top-down
        (0, -90, "XZ"),    # Side
        (0, 0, "YZ")       # Front
    ]

    fig = plt.figure(figsize=(12, 4 * B))

    for i in range(B):
        gt = gt_batch[i]
        pred = pred_batch[i]
        
        for j, (elev, azim, view_name) in enumerate(views):
            ax = fig.add_subplot(B, 3, i * 3 + j + 1, projection='3d')
            ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], c='red', label='GT', s=5)
            ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c='blue', label='Pred', s=5)
            ax.view_init(elev=elev, azim=azim)
            if i == 0:
                ax.set_title(f"{view_name} View")
            if j == 0:
                ax.set_ylabel(f"Sample {i}")
            if j == 2:
                ax.legend(loc='upper right')

    plt.tight_layout()
    wandb.log({"All_Projections": wandb.Image(fig)})
    plt.close(fig)

if __name__ == '__main__':
    test_plot()
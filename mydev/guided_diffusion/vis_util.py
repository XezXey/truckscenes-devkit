import plotly
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import wandb
from plotly.subplots import make_subplots

def plot_distance(pred_batch, gt_batch, name, step):
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
    wandb.log({name: wandb.Image(fig)}, step=int(step))
    plt.close(fig)

    

def plot_2d(pred_batch, gt_batch, name, step):
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
    wandb.log({name: wandb.Image(fig)}, step=int(step))
    plt.close(fig)


# Normalize velocities to unit vectors
def unit_vectors(v):
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    norm[norm==0] = 1
    return v / (norm + 1e-16)

def plot_2d_with_velocity(all_pred, all_gt, name="projection2d", step=0,
                arrow_len=3, wing_len=1):
                #  arrow_len=0.2, wing_len=0.05):
    """
    Simplified plotting with fixed arrow and wing lengths, correctly placing arrows in each subplot.

    Parameters:
    - all_pred, all_gt: np.ndarray of shape (N, 6) [x,y,z,vx,vy,vz]
    - arrow_len: length of each arrow shaft (data units)
    - wing_len: length of each wing segment (data units)
    """

    B = all_pred.shape[0]
    for bi in range(B):
        pred_tmp = all_pred[bi]
        gt_tmp = all_gt[bi]

        # Split positions & velocities
        pts_pred, vel_pred = pred_tmp[:, :3], pred_tmp[:, 3:]
        pts_gt, vel_gt     = gt_tmp[:, :3],   gt_tmp[:, 3:]

        uv_pred = unit_vectors(vel_pred)
        uv_gt   = unit_vectors(vel_gt)

        projections = [(0,1), (0,2), (1,2)]
        titles = ['XY', 'XZ', 'YZ']
        fig = make_subplots(rows=1, cols=3, subplot_titles=titles, horizontal_spacing=0.08)

        def add_arrows(xs, ys, uxs, uys, color, legend_name, legend_group, showlegend, row, col):
            x_coords, y_coords = [], []
            for x0, y0, ux, uy in zip(xs, ys, uxs, uys):
                # Arrow tip
                x_end = x0 + ux * arrow_len
                y_end = y0 + uy * arrow_len
                # Shaft
                x_coords += [x0, x_end, None]
                y_coords += [y0, y_end, None]
                # Base of wings
                bx = x_end - ux * wing_len
                by = y_end - uy * wing_len
                # Perpendicular direction
                px, py = -uy, ux
                # Wing endpoints
                hx1, hy1 = bx + px * wing_len, by + py * wing_len
                hx2, hy2 = bx - px * wing_len, by - py * wing_len
                # Wings
                x_coords += [x_end, hx1, None, x_end, hx2, None]
                y_coords += [y_end, hy1, None, y_end, hy2, None]

            fig.add_trace(
                go.Scatter(
                    x=x_coords, y=y_coords,
                    mode='lines',
                    line=dict(width=1, color=color),
                    name=legend_name,
                    legendgroup=legend_group,
                    showlegend=showlegend
                ),
                row=row, col=col
            )

        for idx, (i, j) in enumerate(projections):
            col = idx + 1
            # Points
            fig.add_trace(go.Scatter(
                x=pts_pred[:, i], y=pts_pred[:, j], mode='markers',
                marker=dict(size=5, color='red'), name='Pred Points',
                legendgroup='PredPts', showlegend=(idx==0)
            ), row=1, col=col)
            fig.add_trace(go.Scatter(
                x=pts_gt[:, i], y=pts_gt[:, j], mode='markers',
                marker=dict(size=5, color='blue'), name='GT Points',
                legendgroup='GTPts', showlegend=(idx==0)
            ), row=1, col=col)

            # Add velocity arrows
            add_arrows(pts_pred[:, i], pts_pred[:, j],
                    uv_pred[:, i], uv_pred[:, j],
                    color='red', legend_name='Pred Velocity',
                    legend_group='PredVel', showlegend=(idx==0),
                    row=1, col=col)
            add_arrows(pts_gt[:, i], pts_gt[:, j],
                    uv_gt[:, i], uv_gt[:, j],
                    color='blue', legend_name='GT Velocity',
                    legend_group='GTVel', showlegend=(idx==0),
                    row=1, col=col)

            # Equal scale
            fig.update_xaxes(
                scaleanchor = f'y{col}',
                scaleratio  = 1,
                row=1, col=col
            )
            # ensure no matching across subplots:
            fig.update_xaxes(matches=None, row=1, col=col)
            fig.update_yaxes(matches=None, row=1, col=col)

        fig.update_layout(
            title=f"{name} @ step {step}",
            legend=dict(x=1.1, y=1),
            margin=dict(l=50, r=200, t=60, b=50),
            width=1400, height=500
        )
        # fig.show()
        html = fig.to_html(include_plotlyjs="cdn")
        wandb.log({name: wandb.Html(html)}, step=int(step))
    # plotly.offline.plot(fig, filename=f'./wandb_vis.html', auto_open=False)
    # wandb.log({name: wandb.Html('./wandb_vis.html')}, step=int(step))



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
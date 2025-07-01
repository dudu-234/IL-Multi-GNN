import torch
import wandb
import logging
from training import get_model
from train_util import AddEgoIds, extract_param, add_arange_ids, get_loaders, evaluate_homo, evaluate_hetero, save_model, load_model
from models import PNA
from mmd import MMD2
from data_loading import get_il_data
from torch_geometric.nn import to_hetero, summary
from torch_geometric.utils import degree

def train_il_gnn(data_old, data_new, args, data_config):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    wandb.init(
        mode="disabled" if args.testing else "online",
        # mode="disabled",
        project="AML_IL_SSRM",
        config={
            "epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "model": args.model,
            "data": args.data,
            "lr": extract_param("lr", args),
            "lambda_mmd": 0.05,
            "n_hidden": extract_param("n_hidden", args),
            "n_gnn_layers": extract_param("n_gnn_layers", args),
            "loss": "ce",
            "w_ce1": extract_param("w_ce1", args),
            "w_ce2": extract_param("w_ce2", args),
            "dropout": extract_param("dropout", args),
            "final_dropout": extract_param("final_dropout", args),
            "n_heads": extract_param("n_heads", args) if args.model == 'gat' else None
        }
    )
    config = wandb.config

    transform = AddEgoIds() if args.ego else None

    # Add unique IDs
    add_arange_ids([data_new])

    tr_loader = get_loaders(data_new, data_new, data_new, None, None, None, transform, args)[0]

    sample_batch = next(iter(tr_loader))    # Build PNA model
    n_feats = sample_batch.x.shape[1]
    e_dim = sample_batch.edge_attr.shape[1] - 1
    model = get_model(sample_batch, config, args)

    model, optimizer = load_model(model, device, args, config, data_config)

    sample_batch.to(device)
    sample_batch.edge_attr = sample_batch.edge_attr[:, 1:]
    logging.info(summary(model, sample_batch.x, sample_batch.edge_index, sample_batch.edge_attr))

    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Precompute Z_edges_old
    model.eval()
    with torch.no_grad():
        Z_nodes_old = model(data_old.x.to(device), data_old.edge_index.to(device), data_old.edge_attr.to(device))
        source_nodes = data_old.edge_index[0]
        target_nodes = data_old.edge_index[1]
        Z_edges_old = torch.cat([Z_nodes_old[source_nodes], Z_nodes_old[target_nodes]], dim=1)

    # Training loop
    model.train()
    for epoch in range(args.n_epochs):
        total_loss = total_task_loss = total_mmd_loss = 0
        for batch in tr_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            output = model(batch.x, batch.edge_index, batch.edge_attr[:, 1:])
            source_nodes = batch.edge_index[0]
            target_nodes = batch.edge_index[1]
            Z_edges_new = torch.cat([output[source_nodes], output[target_nodes]], dim=1)

            loss_mmd = MMD2(Z_edges_old, Z_edges_new, kernel="rbf", device=device)
            loss_task = loss_fn(output.squeeze(), batch.y.float())
            loss = loss_task + 0.05 * loss_mmd

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            total_task_loss += loss_task.item() * batch.num_graphs
            total_mmd_loss += loss_mmd.item() * batch.num_graphs

        avg_loss = total_loss / len(tr_loader.dataset)
        avg_task_loss = total_task_loss / len(tr_loader.dataset)
        avg_mmd_loss = total_mmd_loss / len(tr_loader.dataset)

        wandb.log({"loss/total": avg_loss, "loss/task": avg_task_loss, "loss/mmd": avg_mmd_loss, "epoch": epoch+1})
        logging.info(f"Epoch {epoch+1}/{args.n_epochs} | Loss: {avg_loss:.4f} | Task: {avg_task_loss:.4f} | MMD: {avg_mmd_loss:.4f}")

    # save_path = "checkpoints/pna_aml_il_ssrm.pt"
    # torch.save(model.state_dict(), save_path)
    # logging.info(f"✅ IL + SSRM model saved at {save_path}")

    save_model(model, optimizer, epoch, args, data_config)

    wandb.finish()
    # return model
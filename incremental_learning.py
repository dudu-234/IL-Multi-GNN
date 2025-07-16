from distutils.command import build
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
from torch_geometric.data import Data
from sklearn.cluster import MiniBatchKMeans
import os

def build_edge_emb(h_node, edge_index, edge_attr):
    src, dst = edge_index
    pair = h_node[edge_index.T].reshape(-1, 2 * h_node.size(1)).relu()
    return torch.cat([pair, edge_attr], dim=1)          # (E, 3d)

def build_memory_kmeans(model, data, m=5000, device="cuda"):
    model.eval()
    with torch.no_grad():
        h, _ = model(data.x.to(device),
                     data.edge_index.to(device),
                     data.edge_attr[:, 1:].to(device),
                     return_node_emb=True)

    Z = torch.cat([
        h[data.edge_index[0]],          # h_u
        h[data.edge_index[1]],          # h_v
        data.edge_attr[:, 1:].to(device)  # a_e  
    ], dim=1)                          # (E, 3d)

    km = MiniBatchKMeans(
        n_clusters=m,
        batch_size=4096,
        max_iter=100,
        random_state=42
    )
    km.fit(Z.cpu().numpy())

    centroids = torch.tensor(km.cluster_centers_,
                             device=device,
                             dtype=torch.float32)
    return centroids

def update_memory_kmeans(model, graph, memory, m=5000, device="cuda"):
    model.eval()
    with torch.no_grad():
        h, _ = model(graph.x.to(device),
                     graph.edge_index.to(device),
                     graph.edge_attr[:, 1:].to(device),
                     return_node_emb=True)
    Z_new = torch.cat([
        h[graph.edge_index[0]],
        h[graph.edge_index[1]],
        graph.edge_attr[:, 1:]], dim=1)

    Z_cat = torch.cat([memory, Z_new], dim=0)

    km = MiniBatchKMeans(
            n_clusters=m,
            batch_size=4096,
            max_iter=100,
            random_state=42)
    km.fit(Z_cat.cpu().numpy())
    centroids = torch.tensor(km.cluster_centers_,
                             device=device,
                             dtype=torch.float32)
    return centroids

def concat_graph(g1, g2):
    edge_index = torch.cat([g1.edge_index, g2.edge_index], dim=1)
    edge_attr  = torch.cat([g1.edge_attr,  g2.edge_attr ], dim=0)
    y          = torch.cat([g1.y,          g2.y        ], dim=0)
    return Data(x=g1.x, edge_index=edge_index,
                edge_attr=edge_attr, y=y)


def train_il_gnn(data_old, data_new, args, data_config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

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
            "alpha_mmd": 0.05,
            "beta_mmd": 0.05,
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
    model = get_model(sample_batch, config, args)

    model, optimizer = load_model(model, device, args, config, data_config)

    sample_batch.to(device)
    sample_batch.edge_attr = sample_batch.edge_attr[:, 1:]
    logging.info(summary(model, sample_batch.x, sample_batch.edge_index, sample_batch.edge_attr))

    loss_fn = torch.nn.BCEWithLogitsLoss()

    Z_old = build_memory_kmeans(model, data_old, m=5000, device=device)

    with torch.no_grad():
        h_prev, _ = model(data_old.x.to(device),
                        data_old.edge_index.to(device),
                        data_old.edge_attr[:,1:].to(device),
                        return_node_emb=True)
        Z_prev = build_edge_emb(h_prev,
                                data_old.edge_index.to(device),
                                data_old.edge_attr[:,1:].to(device))

        data_cat = concat_graph(data_old, data_new)
        h_aft, _ = model(data_cat.x.to(device),
                        data_cat.edge_index.to(device),
                        data_cat.edge_attr[:,1:].to(device),
                        return_node_emb=True)
        Z_aft = build_edge_emb(h_aft,
                            data_old.edge_index.to(device),
                            data_old.edge_attr[:,1:].to(device))

    intra_const = MMD2(Z_prev, Z_aft, kernel="rbf", device=device).detach()

    # Training loop
    model.train()
    for epoch in range(args.n_epochs):
        total_loss = total_task_loss = total_mmd_loss = 0
        for batch in tr_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            h_node, logits = model(batch.x,
                                   batch.edge_index,
                                   batch.edge_attr[:, 1:],   # drop Timestamp
                                   return_node_emb=True)

            Z_new = build_edge_emb(h_node, batch.edge_index, batch.edge_attr[:, 1:])

            loss_mmd = MMD2(Z_old, Z_new, kernel="rbf", device=device)
            loss_task = loss_fn(logits.squeeze(), batch.y.float())
            loss = (loss_task + config.alpha_mmd * loss_mmd + config.beta_mmd  * intra_const)

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

    memory = build_memory_kmeans(model, concat_graph(data_old, data_new), m=5000, device=device)

    save_model(model, optimizer, epoch, args, data_config, memory=memory)

    wandb.finish()
    # return model
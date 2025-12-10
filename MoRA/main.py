import argparse
import importlib
import inspect
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import logging
from datetime import datetime

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


from datasets.data import load_multiple
from baselines.model.retriever import IMURetrievalClassifier

# from baselines.RAGMechanism import EXT_CCF_FEATS, EXT_DWT_FEATS, EXT_PEARSON_FEATS
import time

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_model(args):
    name = args.model_name
    camel_name = "".join([i.capitalize() for i in name.split("_")])
    Model = getattr(importlib.import_module(f"baselines.model.{name}"), camel_name)

    class_args = inspect.getfullargspec(Model.__init__).args[1:]
    args_dict = vars(args)
    filtered_args = {arg: args_dict[arg] for arg in class_args if arg in args_dict}
    return Model(**filtered_args)

def feat_reduction(features, output_dim):

    mean = torch.mean(features, dim=0, keepdim=True)
    centered_features = features - mean
    
    cov_matrix = torch.matmul(centered_features.T, centered_features) / (centered_features.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

    idx = torch.argsort(eigenvalues, descending=True)
    eigenvectors = eigenvectors[:, idx]

    top_eigenvectors = eigenvectors[:, :output_dim]
    reduced_features = torch.matmul(centered_features, top_eigenvectors)
    
    return reduced_features

def main(args):
    os.makedirs(args.checkpoint, exist_ok=True)
    log_filename = os.path.join(args.checkpoint, f"{args.model_name}_n-shot={args.n_shot}_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )

    import sys
    cmd = " ".join(sys.argv)
    logging.info(f"Command-line invocation: {cmd}")

    logging.info("=== Starting Experiment ===")
    logging.info(f"Arguments: {args}")

    dataset_list = ['UCI-HAR', 'MotionSense', 'Shoaib', 'RealWorld', 'PAMAP',
                    'USC-HAD', 'WISDM', 'DSADS', 'UTD-MHAD', 'MMAct']
    # dataset_list = ['DSADS']
    dataset_path = "./datasets/downstream"

    train_inputs_list, train_labels_list, label_list_list, all_text_list, num_classes_list = \
        load_multiple(dataset_list, dataset_path, load_method=args.load_method, mode='train', k=args.n_shot)
    test_inputs_list, test_labels_list, _, _, _ = \
        load_multiple(dataset_list, dataset_path, load_method=args.load_method, mode='test')

    for ds, train_inputs, train_labels, test_inputs, test_labels, label_list, all_text, num_classes in \
            zip(dataset_list, train_inputs_list, train_labels_list,
                test_inputs_list, test_labels_list,
                label_list_list, all_text_list, num_classes_list):

        logging.info(f"==== Dataset: {ds} ====")
        args.num_classes = num_classes

        model = load_model(args).to(args.device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.05)

        if args.model_name != "mantis_model":
            backbone_dict = torch.load(os.path.join(args.checkpoint, args.model_name, "backbone_only_weights.pth"), weights_only=True)
            model.backbone.load_state_dict(backbone_dict)

        for _, param in model.backbone.named_parameters():
            param.requires_grad = False

        best_train_loss = float('inf')
        best_imu_ckpt = os.path.join(args.checkpoint, f"{ds}_imu_best.pth")

        start_imu_time = time.time()

        # mask = ~torch.isin(train_labels, torch.tensor([5,6,7,8,9]))
        # train_inputs, train_labels = train_inputs[mask], train_labels[mask]

        train_dataset = TensorDataset(train_inputs, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        for epoch in range(1, args.num_epochs + 1):
            model.train()
            running_loss = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                _, pred = model(inputs)
                loss = F.cross_entropy(pred.float(), labels.long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_dataset)
            logging.info(f"[{ds}] Epoch {epoch} IMU Loss: {epoch_loss:.4f}")

            if epoch_loss < best_train_loss:
                best_train_loss = epoch_loss
                torch.save(model.state_dict(), best_imu_ckpt)

        model.load_state_dict(torch.load(best_imu_ckpt, map_location=args.device))
        model.eval()

        imu_train_time = time.time() - start_imu_time
        logging.info(f"[{ds}] IMU model training time: {imu_train_time:.2f} seconds")

        all_embeddings, all_text_database = [], []
        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs = inputs.to(args.device)
                emb, _ = model(inputs)

                all_embeddings.append(emb.cpu().numpy())
                all_text_database += [label_list[int(i)] for i in labels]
        all_embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
        all_text_database = np.array(all_text_database)

        rag_model = IMURetrievalClassifier(model, args.device, all_embeddings.shape[1], label_list).to(args.device)
        rag_model.build_retrieval_index(all_embeddings, all_text_database)

        optimizer = optim.AdamW(rag_model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.05)
        best_rag_loss = float('inf')
        best_rag_ckpt = os.path.join(args.checkpoint, f"{ds}_rag_best.pth")

        start_rag_time = time.time()

        for epoch in range(1, 1 + 1):
            rag_model.train()
            running_loss = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                output = rag_model(inputs)
                # print(output['beta'])
                loss = rag_model.compute_loss(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_dataset)
            logging.info(f"[{ds}] Epoch {epoch} RAG Loss: {epoch_loss:.4f}")

            if epoch_loss < best_rag_loss:
                best_rag_loss = epoch_loss
                torch.save(rag_model.state_dict(), best_rag_ckpt)

        rag_model.load_state_dict(torch.load(best_rag_ckpt, map_location=args.device))
        rag_model.eval()

        rag_train_time = time.time() - start_rag_time
        logging.info(f"[{ds}] RAG model training time: {rag_train_time:.2f} seconds")

        test_dataset = TensorDataset(test_inputs, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        all_preds = []
        all_rag_preds = []
        all_aug_preds = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(args.device)
                output = rag_model(inputs)

                imu_logits = output['imu_logits'].softmax(-1)
                text_logits = output['text_logits'].softmax(-1)
                final_logits = output['final_logits'].softmax(-1)

                all_preds.append(torch.argmax(imu_logits, dim=-1).cpu().numpy())
                all_rag_preds.append(torch.argmax(args.alpha * text_logits + (1 - args.alpha) * imu_logits,
                                                  dim=-1).cpu().numpy())
                all_aug_preds.append(torch.argmax(final_logits, dim=-1).cpu().numpy())

        pred = np.concatenate(all_preds)
        rag_pred = np.concatenate(all_rag_preds)
        aug_pred = np.concatenate(all_aug_preds)
        acc = accuracy_score(test_labels, pred)
        rag_acc = accuracy_score(test_labels, rag_pred)
        aug_acc = accuracy_score(test_labels, aug_pred)
        f1 = f1_score(test_labels, pred, average='macro', zero_division=0)
        rag_f1 = f1_score(test_labels, rag_pred, average='macro', zero_division=0)
        aug_rag_f1 = f1_score(test_labels, aug_pred, average='macro', zero_division=0)

        logging.info(
            f"{ds} IMU Acc={acc:.4f}, RAG Acc={rag_acc:.4f}, Aug Acc={aug_acc:.4f} | "
            f"F1={f1:.4f}, RAG F1={rag_f1:.4f}, Aug F1={aug_rag_f1:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='One Module Fits All: Retrieval-Augmented Generalization for Motion Time Series')
    parser.add_argument('--seed', type=int, default=42, help='[42 or 3407]')
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--padding_size', type=int, default=200)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--load_method', type=str, default='combine',\
                        help='split or combine')

    parser.add_argument('--model_name', type=str, default='unimts_model',\
                        help='[tslanet/unihar/ts2vec/mantis/imu2clip/unimts/primus/imagebind/onellm/tcn/timesnet_model]')
    parser.add_argument('--stage', type=str, default='without-ft')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4, help='mantis 2e-4 / unimts 1e-4 / ts2vec 1e-3')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--checkpoint', type=str, default='./baselines/checkpoints')

    parser.add_argument('--method', type=str, default='embedding-1', help='correlation-[1-2] / embedding-[1-3]')
    parser.add_argument('--calculation', type=str, default='L2')
    parser.add_argument('--alpha', type=float, default=0.5, help='[0, 0.25, 0.5, 0.75, 1]')
    parser.add_argument('--top_k', type=int, default=5, help='[1, 2, 3, 5, 10]')
    parser.add_argument('--n_shot', type=int, default=None, help='[1, 2, 3, 5, 10]')
    parser.add_argument('--temp', type=int, default=20, help='[1, 5, 10, 20, 25]')

    args = parser.parse_args()
    set_seed(args.seed)
    main(args)

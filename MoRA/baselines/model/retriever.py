import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import faiss
import clip

from ..RAGMechanism import EXT_CCF_FEATS, EXT_DWT_FEATS, EXT_PEARSON_FEATS

class RobustExtractor(nn.Module):
    def __init__(self, local_window_size=10):
        """
        Extracts robust gating features from multi-channel time series data.
        Args:
            local_window_size (int): The size of the local sliding window for local statistics.
        """
        super(RobustExtractor, self).__init__()
        self.local_window_size = local_window_size
        self.norm = nn.LayerNorm(15) # C=6:39 C=3:15 MMAct:

    def forward(self, x):

        # x = x[:,:,:3] if x.shape[2] >=3 else x
        if x.ndim == 5:
            x = x.squeeze(-1)
            # x = x.sum(dim=3)
            nonzero_mask = (x != 0)
            first_nonzero_indices = nonzero_mask.float().argmax(dim=3)
            first_index_value = first_nonzero_indices[0, 0, 0].item()
            x = x[:,:,:,first_index_value]
            x = x.permute(0, 2, 1)

        B, T, C = x.shape
        assert T >= self.local_window_size, "local_window_size must be less than or equal to sequence length"

        # 1. Inter-channel correlation (upper triangle of correlation matrix)
        x_centered = x - x.mean(dim=1, keepdim=True)
        cov_matrix = torch.matmul(x_centered.transpose(1, 2), x_centered) / (T - 1)
        std_dev = torch.std(x, dim=1, keepdim=True)
        std_prod = torch.matmul(std_dev.transpose(1, 2), std_dev)
        corr_matrix = cov_matrix / (std_prod + 1e-8)
        triu_mask = torch.triu(torch.ones(C, C, device=x.device), diagonal=1).bool()
        global_corr = corr_matrix[:, triu_mask]

        # 2. Global variance per channel
        global_variance = torch.var(x, dim=1)

        # 3. Mean absolute difference per channel
        diffs = x[:, 1:] - x[:, :-1]
        global_mad = torch.mean(torch.abs(diffs), dim=1)

        # 4. Zero crossing rate per channel
        sign_changes = torch.sign(x[:, 1:]) != torch.sign(x[:, :-1])
        zcr = sign_changes.float().mean(dim=1)

        # 5. Max local variance range per channel
        x_unfold = x.unfold(1, self.local_window_size, 1)
        local_vars = torch.var(x_unfold, dim=-1)
        max_local_var, _ = local_vars.max(dim=1)
        min_local_var, _ = local_vars.min(dim=1)
        dynamic_range = max_local_var - min_local_var

        # Concatenate all features
        features = torch.cat([global_corr, global_variance, global_mad, zcr, dynamic_range], dim=1)
        return self.norm(features)

class IMURetrieval:
    def __init__(self, device, num_classes, top_k=5, clip_model_name="ViT-B/32"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model, _ = clip.load(clip_model_name, device=self.device)
        self.model.eval()
        self.emb_index = None
        self.phy_index = None
        self.text_cache = {}  # cache for encoded text features
        self.search_cache = {}  # cache for FAISS search results
        self.num_classes = num_classes
        self.top_k = top_k
        faiss.omp_set_num_threads(8)

    def build_index(self, embeddings):
        """Build FAISS index for IMU embeddings, store keys and texts"""
        feature_dim = embeddings.shape[1]
        self.emb_index = faiss.IndexFlatL2(feature_dim)
        self.emb_index.add(embeddings.astype(np.float32))

    def encode_text_cached(self, texts):
        unique_texts = list(set(texts))
        to_encode = [t for t in unique_texts if t not in self.text_cache]
        if to_encode:
            tokens = clip.tokenize(to_encode).to(self.device)
            with torch.no_grad():
                features = self.model.encode_text(tokens)
                features /= features.norm(dim=-1, keepdim=True)
                for t, f in zip(to_encode, features):
                    self.text_cache[t] = f
        return torch.stack([self.text_cache[t] for t in texts]).to(self.device)

    def search(self, query_embeddings, database):
        """Perform batch search using FAISS, return keys and texts"""
        if query_embeddings.shape[0] == 0:
            return []

        if isinstance(query_embeddings, torch.Tensor):
            query_embeddings = query_embeddings.cpu().numpy().astype(np.float32)

        D, I = self.emb_index.search(query_embeddings, self.top_k)
        results = []
        for idx in I:
            results.append(", ".join(database[i] for i in idx))
        return results

    def clip_text_logits(self, ref_features, rag_text, retrieval_used, temperature=20.0):
        batch_size = len(retrieval_used)
        text_logits = torch.zeros(batch_size, self.num_classes, device=self.device)

        if retrieval_used.any():
            valid_indices = retrieval_used.nonzero(as_tuple=True)[0]
            rag_text_valid = [rag_text[i] for i in valid_indices.tolist()]

            with torch.no_grad():
                rag_features = self.encode_text_cached(rag_text_valid)

            logits = temperature * rag_features @ ref_features.T
            probs = logits.softmax(dim=-1)
            text_logits[valid_indices] = probs.float()

        return text_logits

class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.log_sigma1 = nn.Parameter(torch.zeros(1))
        self.log_sigma2 = nn.Parameter(torch.zeros(1))
        self.log_sigma3 = nn.Parameter(torch.zeros(1))

    def forward(self, L_cls, L_align, L_sparse):
        sigma1 = torch.exp(self.log_sigma1)
        sigma2 = torch.exp(self.log_sigma2)
        sigma3 = torch.exp(self.log_sigma3)

        loss = (1 / (2 * sigma1**2)) * L_cls + \
               (1 / (2 * sigma2**2)) * L_align + \
               (1 / (2 * sigma3**2)) * L_sparse + \
               self.log_sigma1 + self.log_sigma2 + self.log_sigma3

        return loss

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

class IMURetrievalClassifier(nn.Module):
    def __init__(self, imu_model, device, hidden_dim=512, label_list=None):
        super().__init__()
        self.device = torch.device(device)

        # Freeze pre-trained IMU model
        self.imu_model = imu_model
        self.imu_model.eval()
        for param in self.imu_model.parameters():
            param.requires_grad = False

        self.num_classes = self.imu_model.num_classes
        self.retriever = IMURetrieval(device, self.num_classes)

        assert label_list is not None, "Must provide label_list during initialization"
        self.label_list = label_list
        self.ref_text_features = self.retriever.encode_text_cached(label_list)

        self.extractor = RobustExtractor()
        # Gating network to decide whether retrieval is needed
        self.switch_net = nn.Sequential(
            nn.BatchNorm1d(15), nn.Linear(15, 128), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(128, 1), nn.Sigmoid()).to(self.device)
        
        self.threshold = nn.Parameter(torch.ones(self.num_classes))
        self.dependency = nn.Parameter(torch.ones(self.num_classes))
        self.loss_function = MultiTaskLoss()

    def build_retrieval_index(self, embeddings, texts):
        """Build retrieval index from IMU embeddings, keys, and texts"""
        self.retriever.build_index(embeddings)
        self.reference_texts = texts

    def extract_imu_features(self, imu_data):
        """Extract features from IMU data using pre-trained model"""
        self.imu_model.eval()
        with torch.no_grad():
            features = self.imu_model(imu_data, return_features=True)
        return features

    def extract_phy_features(self, imu_data):
        """Extracts handcrafted features from multi-channel time series data."""
        return self.extractor(imu_data)

    def forward(self, imu_data):

        with torch.no_grad():
            imu_features, imu_logits = self.imu_model(imu_data)

        phy_features = self.extract_phy_features(imu_data)
        beta = self.switch_net(phy_features).squeeze(-1)

        # imu_features = EXT_CCF_FEATS(imu_data)

        retrieved_texts = self.retriever.search(imu_features, self.reference_texts)
        text_logits = self.retriever.clip_text_logits(
            self.ref_text_features, retrieved_texts, torch.ones_like(beta, dtype=torch.bool))
        
        retrieval_used = beta > 0.5
        # final_logits = imu_logits.clone()
        # final_logits[retrieval_used] = 0.5 * imu_logits[retrieval_used] + 0.5 * text_logits[retrieval_used]
        final_logits = (1 - beta).unsqueeze(-1) * imu_logits + beta.unsqueeze(-1) * text_logits

        return {'imu_logits': imu_logits, 'text_logits': text_logits, 'beta': beta,
            'phy_features': phy_features, 'final_logits': final_logits, 'retrieval_used': retrieval_used}

    def compute_loss(self, outputs, labels):
        """Compute loss for training the gating network"""
        L_cls = F.cross_entropy(outputs['final_logits'].float(), labels.long())

        with torch.no_grad():
            imu_entropy = -F.softmax(outputs['imu_logits'], dim=1) * \
                      F.log_softmax(outputs['imu_logits'], dim=1)
            imu_entropy = imu_entropy.sum(dim=1)

            target_beta = torch.sigmoid(imu_entropy - self.threshold[labels.long()])

        L_align = F.binary_cross_entropy(outputs['beta'], target_beta)

        L_sparse = torch.mean(self.dependency[labels.long()] * outputs['beta'] * (1 - outputs['beta']))

        total_loss = self.loss_function(L_cls, L_align, L_sparse)
        return total_loss

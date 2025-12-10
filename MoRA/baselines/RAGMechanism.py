import torch
import torch.nn.functional as F
import pywt
import scipy.stats
import numpy as np

def EXT_CCF_FEATS(data, max_lag=10):
    # 数据预处理 (保持与原始逻辑一致)
    if data.ndim == 5:
        data = data.squeeze(-1)
        nonzero_mask = (data != 0)
        first_nonzero_indices = nonzero_mask.float().argmax(dim=3)
        first_index_value = first_nonzero_indices[0, 0, 0].item()
        data = data[:,:,:,first_index_value]
        data = data.permute(0, 2, 1)
    
    # 标准化数据 [batch, time, channels]
    data = (data - data.mean(dim=1, keepdim=True)) / (data.std(dim=1, keepdim=True) + 1e-8)
    
    batch_size, n_timesteps, n_channels = data.shape
    n_pairs = n_channels * (n_channels + 1) // 2  # 上三角通道对数
    
    # 生成所有通道对的组合索引
    c1, c2 = torch.triu_indices(n_channels, n_channels)
    
    # 提取所有通道对的数据 [batch, pairs, time]
    x1 = data[:, :, c1]  # shape: [batch, time, pairs]
    x2 = data[:, :, c2]  # shape: [batch, time, pairs]
    x1 = x1.permute(0, 2, 1)  # [batch, pairs, time]
    x2 = x2.permute(0, 2, 1)  # [batch, pairs, time]
    
    # 向量化计算互相关 (使用FFT加速)
    def vectorized_cross_corr(x, y):
        n = x.shape[-1]
        fft_len = 2 * n - 1
        
        # 使用FFT计算互相关
        x_fft = torch.fft.rfft(x, n=fft_len)
        y_fft = torch.fft.rfft(y, n=fft_len)
        corr = torch.fft.irfft(x_fft * y_fft.conj(), n=fft_len)
        
        # 提取非负延迟部分 [0, max_lag]
        return corr[..., :max_lag] / n
    
    # 计算所有通道对的互相关 [batch, pairs, max_lag]
    all_corrs = vectorized_cross_corr(x1, x2)
    
    # 展平特征 [batch, n_pairs * max_lag]
    return all_corrs.reshape(batch_size, -1)

def EXT_DWT_FEATS(data, wavelet='db4', level=3):
    """
    Extract features using Discrete Wavelet Transform (DWT) with batch processing

    Args:
        data: torch.Tensor, input tensor with shape (batch, 200, 6)
        wavelet: str, wavelet type (e.g., 'db1', 'db4', etc.)
        level: int, decomposition level

    Returns:
        features: torch.Tensor, extracted features with shape (batch, feature_dim)
    """
    if data.ndim == 5:
        data = data.squeeze(-1)
        nonzero_mask = (data != 0)
        first_nonzero_indices = nonzero_mask.float().argmax(dim=3)
        first_index_value = first_nonzero_indices[0, 0, 0].item()
        data = data[:, :, :, first_index_value]
        data = data.permute(0, 2, 1)

    batch_size, n_timesteps, n_channels = data.shape
    features_list = []
    
    # 遍历通道 (无法完全向量化)
    for c in range(n_channels):
        channel_data = data[:, :, c].cpu().numpy()
        batch_features = []
        
        # 处理批次中的所有样本
        for sample in channel_data:
            coeffs = pywt.wavedec(sample, wavelet=wavelet, level=level)
            channel_features = []
            
            for coeff in coeffs:
                stats = [
                    coeff.mean(),
                    coeff.std(),
                    (coeff**2).sum(),
                    coeff.max(),
                    scipy.stats.skew(coeff),
                    scipy.stats.kurtosis(coeff)
                ]
                channel_features.extend(stats)
                
            batch_features.append(channel_features)
        
        features_list.append(torch.tensor(batch_features))
    
    # 合并通道特征 [batch, n_features]
    return torch.cat(features_list, dim=1)

def EXT_PEARSON_FEATS(data):
    """
    Extract features using Pearson Correlation Coefficient with batch processing
    
    Args:
        data: torch.Tensor, input data with shape (batch, time, channels)
    
    Returns:
        features: torch.Tensor, extracted features with shape (batch, feature_dim)
    """
    if data.ndim == 5:
        data = data.squeeze(-1)
        nonzero_mask = (data != 0)
        first_nonzero_indices = nonzero_mask.float().argmax(dim=3)
        first_index_value = first_nonzero_indices[0, 0, 0].item()
        data = data[:, :, :, first_index_value]
        data = data.permute(0, 2, 1)
        
    # Normalize data
    batch_size, n_timesteps, n_channels = data.shape
    data = (data - data.mean(dim=1, keepdim=True)) / (data.std(dim=1, keepdim=True) + 1e-8)

    # Compute pairwise correlations
    pearson_features = []
    for c1 in range(n_channels):
        for c2 in range(c1, n_channels):
            x = data[:, :, c1]  # (batch, T)
            y = data[:, :, c2]  # (batch, T)
            corr = torch.mean(x * y, dim=1, keepdim=True)  # (batch, 1)
            pearson_features.append(corr)

    return torch.cat(pearson_features, dim=1)
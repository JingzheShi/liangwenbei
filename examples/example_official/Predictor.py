import os
from .model import *  # 从当前目录下的 model.py 导入所有内容, 使用相对路径
import numpy as np
import pandas as pd
from typing import List

class Predictor():
    def __init__(self):
        self.model = deeplob(num_classes = 3)   
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 指定模型路径，不使用相对路径
        pkl_path = os.path.join(os.path.dirname(__file__), 'best_val_model.pth') 
        state_dicts = torch.load(pkl_path, map_location=self.device) 
        self.model.load_state_dict(state_dicts)
        # 加载模型并移动到对应设备
        self.model.to(self.device)
        self.model.eval()

    def predict(self,x:List[pd.DataFrame])->List[List[int]]:
        with torch.no_grad():
            # 批量预处理所有输入数据
            processed_data = [self.preprocess(df) for df in x]
            processed_data = np.stack([data.values for data in processed_data])  # 转换为 numpy 数组
            x_hat = torch.tensor(processed_data).to(torch.float32).unsqueeze(1).to(self.device)  # (batch_size, 1, height, width)

            # 模型预测
            y = self.model(x_hat)
             
            # 将预测结果移回 CPU
            y = y.cpu().numpy()

            # 生成信号
            y = self.generate_signal(y, single_label=True)
        
            # 确保返回的是List[List[int]]格式
            if isinstance(y[0], list):
                return y
            else:
                # 如果模型输出是单维的，包装成List[List[int]]
                return [y]
    
    ## 这里不重要，预处理方式因人而异
    def preprocess(self, df):
        ''' 数据预处理 '''        
        bid_cols = ['n_bid1', 'n_bid2', 'n_bid3', 'n_bid4', 'n_bid5']
        ask_cols = ['n_ask1', 'n_ask2', 'n_ask3', 'n_ask4', 'n_ask5']
        bsize_cols = ['n_bsize1', 'n_bsize2', 'n_bsize3', 'n_bsize4', 'n_bsize5']
        asize_cols = ['n_asize1', 'n_asize2', 'n_asize3', 'n_asize4', 'n_asize5']

        # 矢量化操作，避免逐列操作
        df[bid_cols] += 1
        df[ask_cols] += 1

        # 使用 NumPy 进行矢量化计算
        bid_values = df[bid_cols].values
        ask_values = df[ask_cols].values
        bsize_values = df[bsize_cols].values
        asize_values = df[asize_cols].values

        # 计算 spread 和 mid_price
        spread = ask_values - bid_values
        mid_price = ask_values + bid_values

        # 计算 weighted_ab
        weighted_ab = (ask_values * bsize_values + bid_values * asize_values) / (bsize_values + asize_values)

        # 计算 vol1_rel_diff 和 volall_rel_diff
        vol1_rel_diff = (bsize_values[:, 0] - asize_values[:, 0]) / (bsize_values[:, 0] + asize_values[:, 0])
        volall_rel_diff = (bsize_values.sum(axis=1) - asize_values.sum(axis=1)) / (bsize_values.sum(axis=1) + asize_values.sum(axis=1))

        # 添加新列
        df['spread1'], df['spread2'], df['spread3'] = spread[:, 0], spread[:, 1], spread[:, 2]
        df['mid_price1'], df['mid_price2'], df['mid_price3'] = mid_price[:, 0], mid_price[:, 1], mid_price[:, 2]
        df['weighted_ab1'], df['weighted_ab2'], df['weighted_ab3'] = weighted_ab[:, 0], weighted_ab[:, 1], weighted_ab[:, 2]
        df['vol1_rel_diff'] = vol1_rel_diff
        df['volall_rel_diff'] = volall_rel_diff

        # 对 amount_delta 应用 np.log1p
        df['amount'] = np.log1p(df['amount_delta'].values)

        # 返回所需的特征列
        feature_col_names = [
            'n_bid1', 'n_bsize1', 'n_bid2', 'n_bsize2', 'n_bid3', 'n_bsize3',
            'n_bid4', 'n_bsize4', 'n_bid5', 'n_bsize5', 'n_ask1', 'n_asize1',
            'n_ask2', 'n_asize2', 'n_ask3', 'n_asize3', 'n_ask4', 'n_asize4',
            'n_ask5', 'n_asize5', 'spread1', 'mid_price1', 'spread2', 'mid_price2',
            'spread3', 'mid_price3', 'weighted_ab1', 'weighted_ab2', 'weighted_ab3',
            'amount', 'vol1_rel_diff', 'volall_rel_diff'
        ]
        return df[feature_col_names]
    
    def generate_signal(self, predict_matrix, class_num=3, single_label=True):
        '''
        Args:
            predict_matrix: np.ndarray 
                - if single_label=True: shape [batch_size, class_num]
                - if single_label=False: shape [batch_size, class_num * label_num]
            class_num: int, number of classes
            single_label: bool, whether each sample has only one label
        Returns:
            signal: List[List[int]]
                - if single_label=True: [[label], [label], ...] (每个样本的标签单独放在一个列表中)
                - if single_label=False: [[label1, label2, ...], ...] (每个样本的多标签列表)
        '''
        if single_label:
            return [[x] for x in predict_matrix.argmax(axis=1)]
        else:
            batch_size = predict_matrix.shape[0]
            label_num = predict_matrix.shape[1] // class_num
            return (
                predict_matrix.reshape(batch_size, class_num, label_num)
                .transpose(0, 2, 1)
                .argmax(axis=2)
                .tolist()
            )


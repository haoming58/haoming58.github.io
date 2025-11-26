"""
移动平均线交易策略
Moving Average Trading Strategy

检测以下信号：
1. 金叉信号：短期均线向上突破长期均线
2. 死叉信号：短期均线向下跌破长期均线
3. 多头排列：MA5、MA10、MA20、MA60依次向上排列
4. 空头排列：MA5、MA10、MA20、MA60依次向下排列
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class MovingAverageStrategy:
    """移动平均线策略类"""
    
    def __init__(self, data: pd.DataFrame):
        """
        初始化策略
        
        参数:
            data: 包含价格数据的DataFrame，需要有'close'列
        """
        self.data = data.copy()
        self.calculate_moving_averages()
    
    def calculate_moving_averages(self):
        """计算各周期移动平均线"""
        self.data['MA5'] = self.data['close'].rolling(window=5).mean()
        self.data['MA10'] = self.data['close'].rolling(window=10).mean()
        self.data['MA20'] = self.data['close'].rolling(window=20).mean()
        self.data['MA60'] = self.data['close'].rolling(window=60).mean()
    
    def detect_golden_cross(self, short_ma: str = 'MA5', long_ma: str = 'MA10') -> pd.Series:
        """
        检测金叉信号
        短周期均线向上突破长周期均线
        
        参数:
            short_ma: 短期均线列名
            long_ma: 长期均线列名
            
        返回:
            布尔序列，True表示出现金叉
        """
        # 当前短期均线在长期均线上方，前一天在下方
        golden_cross = (
            (self.data[short_ma] > self.data[long_ma]) & 
            (self.data[short_ma].shift(1) <= self.data[long_ma].shift(1))
        )
        return golden_cross
    
    def detect_death_cross(self, short_ma: str = 'MA5', long_ma: str = 'MA10') -> pd.Series:
        """
        检测死叉信号
        短周期均线向下跌破长周期均线
        
        参数:
            short_ma: 短期均线列名
            long_ma: 长期均线列名
            
        返回:
            布尔序列，True表示出现死叉
        """
        # 当前短期均线在长期均线下方，前一天在上方
        death_cross = (
            (self.data[short_ma] < self.data[long_ma]) & 
            (self.data[short_ma].shift(1) >= self.data[long_ma].shift(1))
        )
        return death_cross
    
    def detect_price_cross_ma10_up(self) -> pd.Series:
        """
        检测价格从下向上穿越MA10，且MA5在MA10上方
        
        返回:
            布尔序列，True表示满足条件
        """
        price_cross_up = (
            (self.data['close'] > self.data['MA10']) &  # 当前价格在MA10上方
            (self.data['close'].shift(1) <= self.data['MA10'].shift(1)) &  # 前一天价格在MA10下方
            (self.data['MA5'] > self.data['MA10'])  # MA5在MA10上方
        )
        return price_cross_up
    
    def detect_price_cross_ma10_down(self) -> pd.Series:
        """
        检测价格从上向下穿越MA10，且MA5在MA10下方
        
        返回:
            布尔序列，True表示满足条件
        """
        price_cross_down = (
            (self.data['close'] < self.data['MA10']) &  # 当前价格在MA10下方
            (self.data['close'].shift(1) >= self.data['MA10'].shift(1)) &  # 前一天价格在MA10上方
            (self.data['MA5'] < self.data['MA10'])  # MA5在MA10下方
        )
        return price_cross_down
    
    def detect_bullish_alignment(self) -> pd.Series:
        """
        检测多头排列
        MA5 > MA10 > MA20 > MA60，依次向上排列
        
        返回:
            布尔序列，True表示多头排列
        """
        bullish = (
            (self.data['MA5'] > self.data['MA10']) &
            (self.data['MA10'] > self.data['MA20']) &
            (self.data['MA20'] > self.data['MA60'])
        )
        return bullish
    
    def detect_bearish_alignment(self) -> pd.Series:
        """
        检测空头排列
        MA5 < MA10 < MA20 < MA60，依次向下排列
        
        返回:
            布尔序列，True表示空头排列
        """
        bearish = (
            (self.data['MA5'] < self.data['MA10']) &
            (self.data['MA10'] < self.data['MA20']) &
            (self.data['MA20'] < self.data['MA60'])
        )
        return bearish
    
    def get_all_signals(self) -> pd.DataFrame:
        """
        获取所有交易信号
        
        返回:
            包含所有信号的DataFrame
        """
        signals = pd.DataFrame(index=self.data.index)
        
        # 金叉和死叉信号
        signals['golden_cross_5_10'] = self.detect_golden_cross('MA5', 'MA10')
        signals['golden_cross_10_20'] = self.detect_golden_cross('MA10', 'MA20')
        signals['golden_cross_5_20'] = self.detect_golden_cross('MA5', 'MA20')
        
        signals['death_cross_5_10'] = self.detect_death_cross('MA5', 'MA10')
        signals['death_cross_10_20'] = self.detect_death_cross('MA10', 'MA20')
        signals['death_cross_5_20'] = self.detect_death_cross('MA5', 'MA20')
        
        # 价格穿越信号
        signals['price_cross_ma10_up'] = self.detect_price_cross_ma10_up()
        signals['price_cross_ma10_down'] = self.detect_price_cross_ma10_down()
        
        # 均线排列信号
        signals['bullish_alignment'] = self.detect_bullish_alignment()
        signals['bearish_alignment'] = self.detect_bearish_alignment()
        
        return signals
    
    def get_comprehensive_signal(self) -> pd.Series:
        """
        获取综合交易信号
        
        返回:
            1: 强烈买入
            0.5: 买入
            0: 中性
            -0.5: 卖出
            -1: 强烈卖出
        """
        signals = self.get_all_signals()
        score = pd.Series(0.0, index=self.data.index)
        
        # 多头信号加分
        score += signals['golden_cross_5_10'] * 0.3
        score += signals['golden_cross_10_20'] * 0.4
        score += signals['price_cross_ma10_up'] * 0.3
        score += signals['bullish_alignment'] * 0.5
        
        # 空头信号减分
        score -= signals['death_cross_5_10'] * 0.3
        score -= signals['death_cross_10_20'] * 0.4
        score -= signals['price_cross_ma10_down'] * 0.3
        score -= signals['bearish_alignment'] * 0.5
        
        return score
    
    def print_latest_signals(self, n: int = 5):
        """
        打印最近n天的信号
        
        参数:
            n: 显示最近多少天的数据
        """
        signals = self.get_all_signals()
        score = self.get_comprehensive_signal()
        
        print(f"\n最近{n}天的交易信号：")
        print("=" * 100)
        
        for i in range(-n, 0):
            if abs(i) > len(self.data):
                continue
                
            date = self.data.index[i]
            print(f"\n日期: {date}")
            print(f"收盘价: {self.data['close'].iloc[i]:.2f}")
            print(f"MA5: {self.data['MA5'].iloc[i]:.2f}, MA10: {self.data['MA10'].iloc[i]:.2f}, "
                  f"MA20: {self.data['MA20'].iloc[i]:.2f}, MA60: {self.data['MA60'].iloc[i]:.2f}")
            
            # 显示触发的信号
            active_signals = []
            for col in signals.columns:
                if signals[col].iloc[i]:
                    active_signals.append(col)
            
            if active_signals:
                print(f"触发信号: {', '.join(active_signals)}")
            else:
                print("触发信号: 无")
            
            print(f"综合评分: {score.iloc[i]:.2f}")
            
            # 给出建议
            if score.iloc[i] >= 0.8:
                print("建议: 强烈买入 🚀")
            elif score.iloc[i] >= 0.3:
                print("建议: 买入 📈")
            elif score.iloc[i] <= -0.8:
                print("建议: 强烈卖出 ⚠️")
            elif score.iloc[i] <= -0.3:
                print("建议: 卖出 📉")
            else:
                print("建议: 观望 ⏸️")


def example_usage():
    """示例用法"""
    # 创建示例数据
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # 生成模拟价格数据（带趋势）
    trend = np.linspace(100, 120, 100)
    noise = np.random.randn(100) * 2
    prices = trend + noise
    
    df = pd.DataFrame({
        'close': prices
    }, index=dates)
    
    # 初始化策略
    strategy = MovingAverageStrategy(df)
    
    # 获取所有信号
    all_signals = strategy.get_all_signals()
    
    # 打印最近5天的信号
    strategy.print_latest_signals(n=5)
    
    # 统计各类信号出现次数
    print("\n\n信号统计：")
    print("=" * 50)
    for col in all_signals.columns:
        count = all_signals[col].sum()
        if count > 0:
            print(f"{col}: {count}次")


if __name__ == "__main__":
    example_usage()

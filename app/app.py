import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import os
"""
http://127.0.0.1:8050
"""
# 路径
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
NODE_COORDS_CSV = os.path.join(DATA_DIR, 'node_coords.csv')
PRED_PATH = os.path.join(DATA_DIR, 'predictions.npy')
TRUE_PATH = os.path.join(DATA_DIR, 'y_true.npy')

# 读取节点坐标
node_df = pd.read_csv(NODE_COORDS_CSV)

# 读取预测和真值（如果存在）
try:
    y_pred = np.load(PRED_PATH)  # (样本数, 节点数, 预测步)
    y_true = np.load(TRUE_PATH)
    has_data = True
except FileNotFoundError:
    print("预测数据文件不存在，请先运行train.py生成数据")
    y_pred = np.random.randn(100, len(node_df), 12)  # 模拟数据
    y_true = np.random.randn(100, len(node_df), 12)
    has_data = False

app = dash.Dash(__name__)

# 自定义CSS样式
app.layout = html.Div([
    # 页面头部
    html.Div([
        html.H1('🚦 PEMS08 交通流预测可视化系统', 
                style={
                    'textAlign': 'center',
                    'color': '#2c3e50',
                    'marginBottom': '10px',
                    'fontFamily': 'Arial, sans-serif',
                    'fontWeight': 'bold'
                }),
        html.Div([
            html.Span('📊 数据状态: ', style={'fontWeight': 'bold', 'color': '#34495e'}),
            html.Span(
                '✅ 真实预测数据' if has_data else '🔄 模拟数据（请先运行train.py）',
                style={
                    'color': '#27ae60' if has_data else '#e74c3c',
                    'fontWeight': 'bold',
                    'backgroundColor': '#ecf0f1',
                    'padding': '5px 10px',
                    'borderRadius': '15px'
                }
            )
        ], style={'textAlign': 'center', 'marginBottom': '20px'})
    ], style={
        'backgroundColor': '#f8f9fa',
        'padding': '20px',
        'borderRadius': '10px',
        'marginBottom': '20px',
        'boxShadow': '0 2px 10px rgba(0,0,0,0.1)'
    }),
    
    # 主要内容区域
    html.Div([
        # 左侧：节点分布图
        html.Div([
            html.H3('🗺️ 节点网络分布图', 
                   style={'color': '#2c3e50', 'marginBottom': '15px', 'textAlign': 'center'}),
            dcc.Graph(
                id='node-map',
                figure=go.Figure(
                    data=[go.Scatter(
                        x=node_df['x'], y=node_df['y'], 
                        mode='markers', 
                        marker=dict(
                            size=10, 
                            color='#3498db',
                            line=dict(width=2, color='#2980b9'),
                            opacity=0.8
                        ), 
                        text=node_df['node_id'], 
                        customdata=node_df['node_id'], 
                        name='传感器节点',
                        hovertemplate='<b>节点ID: %{text}</b><br>' +
                                    'X坐标: %{x:.3f}<br>' +
                                    'Y坐标: %{y:.3f}<br>' +
                                    '<i>点击查看预测曲线</i><extra></extra>'
                    )],   # type: ignore
                    layout=go.Layout(
                        title={
                            'text': '基于距离关系的网络拓扑布局',
                            'x': 0.5,
                            'font': {'size': 14, 'color': '#34495e'}
                        },
                        clickmode='event+select',
                        xaxis=dict(title='X坐标', gridcolor='#ecf0f1'),
                        yaxis=dict(title='Y坐标', gridcolor='#ecf0f1'),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(family='Arial, sans-serif')
                    )   # type: ignore
                ),   # type: ignore
                style={'height': '600px', 'border': '1px solid #bdc3c7', 'borderRadius': '8px'}
            )
        ], style={
            'width': '48%', 
            'display': 'inline-block', 
            'verticalAlign': 'top',
            'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '10px',
            'boxShadow': '0 2px 10px rgba(0,0,0,0.1)',
            'marginRight': '2%'
        }),
        
        # 右侧：预测曲线图
        html.Div([
            html.H3('📈 节点预测曲线', 
                   style={'color': '#2c3e50', 'marginBottom': '15px', 'textAlign': 'center'}),
            html.Div(id='node-info', 
                    style={
                        'textAlign': 'center', 
                        'fontSize': '16px',
                        'color': '#7f8c8d',
                        'marginBottom': '15px',
                        'padding': '10px',
                        'backgroundColor': '#ecf0f1',
                        'borderRadius': '8px',
                        'fontStyle': 'italic'
                    }),
            dcc.Graph(id='node-prediction', 
                     style={'height': '500px', 'border': '1px solid #bdc3c7', 'borderRadius': '8px'})
        ], style={
            'width': '48%', 
            'display': 'inline-block', 
            'verticalAlign': 'top',
            'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '10px',
            'boxShadow': '0 2px 10px rgba(0,0,0,0.1)'
        })
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'flex-start'}),
    
    # 页面底部
    html.Div([
        html.P('💡 使用说明：点击左侧节点分布图中的任意节点，右侧将显示该节点的交通流预测与真实值对比曲线',
               style={
                   'textAlign': 'center',
                   'color': '#7f8c8d',
                   'fontStyle': 'italic',
                   'marginTop': '20px'
               })
    ])
], style={
    'fontFamily': 'Arial, sans-serif',
    'backgroundColor': '#f5f6fa',
    'padding': '20px',
    'minHeight': '100vh'
})

@app.callback(
    [Output('node-prediction', 'figure'), Output('node-info', 'children')],
    Input('node-map', 'clickData')
)
def update_node_prediction(clickData):
    if clickData is None:
        empty_fig = go.Figure()   # type: ignore
        empty_fig.update_layout(
            title='等待选择节点...',
            xaxis_title='样本序号',
            yaxis_title='交通流量',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Arial, sans-serif', color='#7f8c8d')
        )
        return empty_fig, '👆 点击左侧地图中的节点查看预测曲线'
    
    # 修复数据获取方式
    try:
        if 'customdata' in clickData['points'][0]:
            node_id = int(clickData['points'][0]['customdata'])
        else:
            node_id = int(clickData['points'][0]['pointIndex'])
    except (KeyError, IndexError, ValueError):
        return go.Figure(), '❌ 数据获取错误，请重新点击节点'   # type: ignore
    
    # 确保节点ID在有效范围内
    if node_id >= len(node_df):
        return go.Figure(), f'❌ 节点ID {node_id} 超出范围'   # type: ignore
    
    # 创建美化的预测图表
    fig = go.Figure()   # type: ignore
    
    # 获取预测步长
    pred_len = y_true.shape[2]
    
    # 只显示第12步的预测（最后一个预测步长）
    final_step = pred_len - 1  # 第12步（索引为11）
    
    # 真实值曲线
    fig.add_trace(go.Scatter(
        y=y_true[:, node_id, final_step], 
        name='🔵 真实值', 
        line=dict(color='#3498db', width=3),
        hovertemplate='真实值: %{y:.2f}<extra></extra>'
    ))   # type: ignore
    
    # 预测值曲线
    fig.add_trace(go.Scatter(
        y=y_pred[:, node_id, final_step], 
        name='🔴 预测值', 
        line=dict(color='#e74c3c', width=3, dash='dot'),
        hovertemplate='预测值: %{y:.2f}<extra></extra>'
    ))   # type: ignore
    
    # 美化图表布局
    fig.update_layout(
        title={
            'text': f'节点 {node_id} 的交通流预测对比',
            'x': 0.5,
            'font': {'size': 16, 'color': '#2c3e50'}
        },
        xaxis=dict(
            title='时间步长',
            gridcolor='#ecf0f1',
            showgrid=True
        ),
        yaxis=dict(
            title='交通流量',
            gridcolor='#ecf0f1',
            showgrid=True
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            x=0.02, y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#bdc3c7',
            borderwidth=1
        ),
        font=dict(family='Arial, sans-serif'),
        hovermode='x unified'
    )
    
    # 节点信息
    coord_x = node_df.iloc[node_id]["x"]
    coord_y = node_df.iloc[node_id]["y"]
    info = f'🎯 当前节点: {node_id} | 📍 坐标: ({coord_x:.3f}, {coord_y:.3f}) | 预测步长: {pred_len}'
    
    return fig, info

if __name__ == '__main__':
    app.run(debug=True) 
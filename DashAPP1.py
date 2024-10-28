import pandas as pd
import numpy as np
import plotly.express as px
from dash import dcc, html
from dash.dependencies import Input, Output


# 权重字典
weights = {
    'Administration': {
        'Leadership': 0.20, 'Communication Skills': 0.20, 'Emotional Intelligence': 0.15, 'Decision-Making': 0.15, 'Teamwork': 0.10,
        'Education Level': 0.10, 'Problem-Solving Skills': 0.05, 'Creativity': 0.05, 'Technical Proficiency': 0.05, 'Self-Motivation': 0.05
    },
    'Sales': {
        'Communication Skills': 0.25, 'Emotional Intelligence': 0.20, 'Self-Motivation': 0.20, 'Creativity': 0.10, 'Teamwork': 0.10,
        'Education Level': 0.05, 'Problem-Solving Skills': 0.05, 'Leadership': 0.05, 'Technical Proficiency': 0, 'Decision-Making': 0
    },
    'Development': {
        'Technical Proficiency': 0.25, 'Problem-Solving Skills': 0.20, 'Creativity': 0.15, 'Teamwork': 0.10, 'Decision-Making': 0.10,
        'Education Level': 0.05, 'Emotional Intelligence': 0.05, 'Leadership': 0.05, 'Communication Skills': 0.05, 'Self-Motivation': 0
    },
    'Data Science': {
        'Technical Proficiency': 0.25, 'Problem-Solving Skills': 0.20, 'Creativity': 0.15, 'Education Level': 0.15, 'Decision-Making': 0.10,
        'Teamwork': 0.05, 'Emotional Intelligence': 0.05, 'Leadership': 0.05, 'Communication Skills': 0.05, 'Self-Motivation': 0.05
    },
    'Research & Development': {
        'Technical Proficiency': 0.25, 'Problem-Solving Skills': 0.20, 'Creativity': 0.20, 'Teamwork': 0.10, 'Education Level': 0.10,
        'Decision-Making': 0.05, 'Emotional Intelligence': 0.05, 'Leadership': 0.05, 'Communication Skills': 0.05, 'Self-Motivation': 0.05
    }
}

# 定义计算加权得分的函数
def calculate_weighted_scores(employee_data):
    scores = {}
    for department, weight_dict in weights.items():
        total_score = 0
        for skill, weight in weight_dict.items():
            total_score += employee_data[skill] * weight
        scores[department] = total_score
    return scores

# 更新 Dashboard 的 callback 函数
@app.callback(
    [Output('monthly-performance', 'figure'),
     Output('weighted-scores', 'figure'),
     Output('prediction-output', 'children'),
     Output('employee-info-table', 'children')],
    [Input('employee-dropdown', 'value'),
     Input('employee-id-input', 'value')]
)
def update_dashboard(dropdown_value, input_value):
    # 获取员工ID
    employee_id = input_value if input_value is not None else dropdown_value
    employee_data = data[data['employee_id'] == employee_id]

    # 检查是否存在数据
    if employee_data.empty:
        return {}, {}, "Employee ID not found.", "No data available for this Employee ID."

    try:
        # 提取每月的绩效图表数据
        fig_bar = px.bar(employee_data, x="date", y="Score", title=f"Performance Scores for Employee {employee_id}",
                         template="plotly_white", color="Performance Metric")

        # 计算每个部门的加权得分
        latest_employee_data = employee_data.drop(columns=['employee_id', 'date', 'current_department']).iloc[-1]
        weighted_scores = calculate_weighted_scores(latest_employee_data)

        # 绘制加权得分图表
        fig_weighted_scores = px.bar(x=list(weighted_scores.keys()), y=list(weighted_scores.values()),
                                     title=f"Weighted Scores across Departments for Employee {employee_id}",
                                     labels={'x': "Department", 'y': "Weighted Score"},
                                     template="plotly_white")

        # 预测适合部门的文本信息
        best_fit_department = max(weighted_scores, key=weighted_scores.get)
        prediction_text = f"The best-fit department for Employee {employee_id} based on weighted scores is: {best_fit_department}"

        # 员工信息表
        employee_info_table = html.Table([
            html.Tr([html.Th("Attribute"), html.Th("Value")]),
            html.Tr([html.Td("Employee ID"), html.Td(employee_id)]),
            html.Tr([html.Td("Current Department"), html.Td(employee_data['current_department'].iloc[0])])
        ], style={'width': '100%', 'border': '1px solid #d1d1d6', 'border-collapse': 'collapse', 'border-radius': '8px',
                  'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.05)'})

    except Exception as e:
        # 捕获异常并返回默认值
        print(f"Exception encountered: {e}")
        return {}, {}, "Error in processing the data.", "Error in generating the employee information table."

    # 返回生成的图表和信息
    return fig_bar, fig_weighted_scores, prediction_text, employee_info_table

if __name__ == '__main__':
    app.run_server(debug=True)
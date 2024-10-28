import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

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

# 计算加权得分的函数
def calculate_weighted_scores(employee_data):
    scores = {}
    for department, weight_dict in weights.items():
        total_score = 0
        for skill, weight in weight_dict.items():
            total_score += employee_data[skill] * weight
        scores[department] = total_score
    return scores

# Load data
data = pd.read_csv('employee_monthly_data.csv')

# Define the prediction function
def generate_predictions(employee_features):
    # Load the trained random forest model
    model = joblib.load("random_forest_model2.pkl")

    # Define the feature names used in training
    feature_names = ['Education Level', 'Leadership', 'Communication Skills', 'Self-Motivation',
                     'Problem-Solving Skills', 'Technical Proficiency', 'Emotional Intelligence',
                     'Creativity', 'Teamwork', 'Decision-Making']

    # Ensure employee_features is a DataFrame with the correct feature names
    employee_features_df = pd.DataFrame([employee_features], columns=feature_names)

    # Get probability scores and prediction
    prediction_proba = model.predict_proba(employee_features_df)[0]
    prediction = np.argmax(prediction_proba)

    # Define department mapping
    departments = ['Administration', 'Sales', 'Development', 'Data Science', 'Research & Development']
    predicted_department = departments[prediction]
    predicted_score = prediction_proba[prediction] * 100  # Prediction score as a percentage
    is_qualified = "Yes" if predicted_score >= 70 else "No"  # Qualification threshold

    return predicted_department, predicted_score, is_qualified

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

# Layout
app.layout = html.Div(style={'backgroundColor': '#f5f5f7', 'padding': '30px',
                             'font-family': '-apple-system, BlinkMacSystemFont, sans-serif'}, children=[
    html.H1("Employee Performance Dashboard", style={'textAlign': 'center', 'color': '#1c1c1e'}),

    # Employee selection, prediction output, and employee info side-by-side at the top
    html.Div([
        # Employee selection on the left
        html.Div([
            html.Label("Select Employee", style={'font-weight': '600', 'color': '#1c1c1e'}),
            dcc.Dropdown(
                id='employee-dropdown',
                options=[{'label': f"Employee {i}", 'value': i} for i in data['employee_id'].unique()],
                value=data['employee_id'].iloc[0],
                style={'width': '100%', 'margin-bottom': '10px', 'border-radius': '8px', 'padding': '5px',
                       'border': '1px solid #d1d1d6'}
            ),
            html.Label("Or Enter Employee ID", style={'font-weight': '600', 'color': '#1c1c1e'}),
            dcc.Input(id='employee-id-input', type='number', placeholder="Enter Employee ID",
                      style={'width': '100%', 'padding': '10px', 'border-radius': '8px', 'border': '1px solid #d1d1d6',
                             'margin-top': '10px'})
        ], style={'backgroundColor': '#ffffff', 'padding': '20px', 'border-radius': '12px',
                  'box-shadow': '0 4px 10px rgba(0, 0, 0, 0.05)', 'width': '30%', 'margin-right': '10px'}),

        # Prediction output in the middle
        html.Div(id='prediction-output', style={'padding': '20px', 'font-size': '20px', 'textAlign': 'center',
                                                'backgroundColor': '#ffffff', 'border-radius': '12px',
                                                'box-shadow': '0 4px 10px rgba(0, 0, 0, 0.05)', 'width': '30%', 'margin': '0 10px'}),

        # Employee info table on the right
        html.Div(id='employee-info-table', style={'backgroundColor': '#ffffff', 'padding': '20px',
                                                  'border-radius': '12px', 'box-shadow': '0 4px 10px rgba(0, 0, 0, 0.05)',
                                                  'font-size': '18px', 'color': '#1c1c1e', 'width': '30%', 'margin-left': '10px'})
    ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-bottom': '20px'}),

    # Chart output
    html.Div([
        dcc.Graph(id='monthly-performance'),
        dcc.Graph(id='monthly-trend'),
        dcc.Graph(id='weighted-scores')
    ], style={'backgroundColor': '#ffffff', 'padding': '20px', 'border-radius': '12px',
              'box-shadow': '0 4px 10px rgba(0, 0, 0, 0.05)', 'margin': '20px auto', 'width': '80%'})
])




# Callback function
@app.callback(
    [Output('monthly-performance', 'figure'),
     Output('monthly-trend', 'figure'),
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
        return {}, {}, {}, "Employee ID not found.", "No data available for this Employee ID."

    try:
        # 计算每个月的加权得分
        monthly_scores = []
        for _, row in employee_data.iterrows():
            scores = calculate_weighted_scores(row)
            scores['date'] = row['date']  # 保留日期信息
            monthly_scores.append(scores)

        # 转换为 DataFrame
        monthly_scores_df = pd.DataFrame(monthly_scores)

        # 将数据转换为长格式，以便生成折线图
        monthly_scores_long = monthly_scores_df.melt(id_vars=['date'], var_name='Department', value_name='Weighted Score')

        # 每月加权得分的折线图
        fig_line = px.line(monthly_scores_long, x="date", y="Weighted Score", color="Department",
                           title=f"Monthly Weighted Scores by Department for Employee {employee_id}",
                           labels={'date': 'Date', 'Weighted Score': 'Score'},
                           template="plotly_white")

        # 从 employee_data 中提取特征列，用于预测
        employee_features = employee_data.drop(columns=['employee_id', 'date', 'current_department']).iloc[0].values

        # 预测结果
        predicted_department, predicted_score, is_qualified = generate_predictions(employee_features)
        prediction_text = html.Div([
            html.Span(f"Predicted Suitable Department for Employee {employee_id}: {predicted_department}"),
            html.Br(),  # 换行
            html.Span(f"Prediction Score: {predicted_score:.2f}% | Qualified: {is_qualified}",
                      style={'font-weight': 'bold', 'color': 'red'})
        ])

        # 计算最新月的加权得分
        latest_employee_data = employee_data.drop(columns=['employee_id', 'date', 'current_department']).iloc[-1]
        weighted_scores = calculate_weighted_scores(latest_employee_data)

        # 加权得分柱状图
        fig_weighted_scores = px.bar(x=list(weighted_scores.keys()), y=list(weighted_scores.values()),
                                     title=f"Weighted Scores across Departments for Employee {employee_id}",
                                     labels={'x': "Department", 'y': "Weighted Score"},
                                     template="plotly_white")

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
        return {}, {}, {}, "Error in processing the data.", "Error in generating the employee information table."

    # 返回生成的图表和信息，包括加权得分图表
    return fig_line, fig_line, fig_weighted_scores, prediction_text, employee_info_table


if __name__ == '__main__':
    app.run_server(debug=True)
















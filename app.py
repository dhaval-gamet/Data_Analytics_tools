import os
from flask import Flask, request, jsonify, render_template, send_file
from data_analysis import DataAnalysisTool
import tempfile

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')

# Initialize the data analysis tool
data_tool = DataAnalysisTool()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'})
    
    success, message = data_tool.load_data(file, file.filename)
    
    if success:
        suggestions = data_tool.auto_data_cleaning_suggestions()
        return jsonify({
            'success': True, 
            'message': message,
            'suggestions': suggestions,
            'shape': data_tool.df.shape if data_tool.df is not None else (0, 0)
        })
    else:
        return jsonify({'success': False, 'message': message})

@app.route('/analyze', methods=['GET'])
def analyze_data():
    analysis_result = data_tool.analyze_data()
    if analysis_result:
        report = data_tool.display_analysis_results()
        return jsonify({'success': True, 'report': report})
    else:
        return jsonify({'success': False, 'message': 'No data to analyze'})

@app.route('/clean', methods=['POST'])
def clean_data():
    strategies = request.json.get('strategies', None)
    success, message = data_tool.clean_data(strategies)
    return jsonify({'success': success, 'message': message})

@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    recommendations = data_tool.recommend_charts()
    return jsonify({'success': True, 'recommendations': recommendations})

@app.route('/advanced_analysis', methods=['GET'])
def advanced_analysis():
    result, message = data_tool.perform_advanced_analysis()
    if result is not None:
        # Save result to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            result.to_csv(f.name, index=False)
            temp_file = f.name
        
        return send_file(temp_file, as_attachment=True, download_name='advanced_analysis_result.csv')
    else:
        return jsonify({'success': False, 'message': message})

@app.route('/generate_report', methods=['GET'])
def generate_report():
    success, result = data_tool.generate_automated_report()
    if success:
        return send_file(result, as_attachment=True)
    else:
        return jsonify({'success': False, 'message': result})

@app.route('/preview', methods=['GET'])
def preview_data():
    if data_tool.df is not None:
        preview = data_tool.df.head(10).to_html(classes='table table-striped')
        shape = data_tool.df.shape
        missing = data_tool.df.isnull().sum().sum()
        duplicates = data_tool.df.duplicated().sum()
        
        return jsonify({
            'success': True,
            'preview': preview,
            'shape': f"{shape[0]} rows, {shape[1]} columns",
            'missing': f"{missing} missing values",
            'duplicates': f"{duplicates} duplicate rows"
        })
    else:
        return jsonify({'success': False, 'message': 'No data loaded'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('DEBUG', False))

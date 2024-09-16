
from xgboost import XGBClassifier
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from flask import Flask, request, render_template, redirect, url_for, session
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import csv

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Add a secret key for session management

# Load and preprocess the dataset
data = pd.read_csv('data/Employee_Attrition.csv')

le_dict = {}
for column in data.columns:
    if data[column].dtype == object:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        le_dict[column] = le 

# Define features and target
identifiers = data['EmployeeNumber']
departments = data['Department']
X = data.drop(['Attrition', 'EmployeeNumber', 'Department'], axis=1)
y = data['Attrition']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost classifier and train the model
xgb_model = XGBClassifier(eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Function to read user data from CSV
def read_user_data(filename):
    user_data = {}
    try:
        with open(filename, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                user_data[row['username']] = row['password']
    except FileNotFoundError:
        pass
    return user_data

# Function to save user data to CSV
def save_user_data(filename, username, password):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([username, password])

# Function to authenticate user
def authenticate_user(username, password, user_data):
    return user_data.get(username) == password

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            return render_template('register.html', error="Passwords do not match")

        user_data = read_user_data('user.csv')
        if username in user_data:
            return render_template('register.html', error="Username already exists")

        save_user_data('user.csv', username, password)
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user_data = read_user_data('user.csv')
        if authenticate_user(username, password, user_data):
            session['username'] = username
            return redirect(url_for('layoff_prediction'))  # Redirect to index after login
        else:
            return render_template('login.html', error="Invalid username or password")

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/individual', methods=['GET', 'POST'])
def individual():
    if request.method == 'POST':
        employee_number = int(request.form['employeeNumber'])
        employee_details = data[data['EmployeeNumber'] == employee_number]

        if employee_details.empty:
            return render_template('result.html', result=f'Employee number {employee_number} not found.', skills=None)

        employee_department = employee_details['Department'].values[0]
        employee_department_new = le_dict['Department'].inverse_transform([employee_department])[0]
        employee_job_role = employee_details['JobRole'].values[0]
        employee_details = employee_details.drop(['EmployeeNumber', 'Attrition', 'Department'], axis=1)
        prediction = xgb_model.predict(employee_details)
        if prediction[0] == 1:
            result = 'Laid off'
            skills = recommend_skills(employee_department_new, employee_job_role)
            skill_list = ", ".join(skills)
            return render_template('result.html', result=f'Employee number {employee_number} is predicted to be: {result}.', skills=skill_list)
        else:
            result = 'Not laid off'
            return render_template('result.html', result=f'Employee number {employee_number} is predicted to be: {result}.', skills=None)

    return render_template('individual.html')

@app.route('/job_shift')
def job_shift():
    return render_template('job_shift.html')


@app.route('/layoff_prediction')
def layoff_prediction():
    if 'username' not in session:
        return redirect(url_for('login'))

    # Make predictions on the test set
    y_pred = xgb_model.predict(X_test)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Convert the confusion matrix to a DataFrame for better visualization
    cm_df = pd.DataFrame(cm, index=['Actual Not Laid Off', 'Actual Laid Off'],
                         columns=['Predicted Not Laid Off', 'Predicted Laid Off'])
    # Plot confusion matrix
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()

    # Save plot to a BytesIO object and encode as base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    confusion_matrix_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()

    # Visualization for total employees and predicted layoffs
    y_pred_all = xgb_model.predict(X)
    results = pd.DataFrame({'Identifier': identifiers, 'Department': departments, 'Actual': y, 'Predicted': y_pred_all})
    laid_off_employees = results[results['Predicted'] == 1]

    # Convert department codes back to names
    results['Department'] = le_dict['Department'].inverse_transform(results['Department'])
    laid_off_employees['Department'] = le_dict['Department'].inverse_transform(laid_off_employees['Department'])

    # Total employees and predicted layoffs by department
    plt.figure(figsize=(14, 8))
    visualization_df = pd.DataFrame({
        'Total Employees': results['Department'].value_counts(),
        'Predicted Layoffs': laid_off_employees['Department'].value_counts()
    }).fillna(0)
    visualization_df.plot(kind='bar')
    plt.title('Total Employees and Predicted Layoffs by Department')
    plt.xlabel('Department')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()

    # Number of layoffs plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=laid_off_employees['Department'].value_counts().index,
                y=laid_off_employees['Department'].value_counts().values)
    plt.title('Number of Predicted Layoffs by Department')
    plt.xlabel('Department')
    plt.ylabel('Number of Layoffs')
    plt.xticks(rotation=45)
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    layoffs_plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()

    return render_template(
        'layoff_prediction.html',
        plot_url=plot_url,
        layoffs_plot_url=layoffs_plot_url,
        confusion_matrix_url=confusion_matrix_url,
        laid_off_employees=laid_off_employees.to_html(classes='table table-striped')
    
    )



def recommend_skills(department, job_role):
    skills = {
        "Human Resources": ["Employee Relations", "Talent Acquisition", "Conflict Resolution"],
        "Engineering": ["Data Analysis", "Machine Learning", "Software Development"],
        "Sales": ["Customer Relationship Management", "Sales Strategy", "Negotiation Skills"],
        "Marketing": ["Digital Marketing", "SEO/SEM", "Content Creation"],
        "Research & Development":["Data Analysis","Experimental Design","Technical Writing"],
        "Finance": ["Financial Analysis", "Accounting", "Budgeting and Forecasting"],
        "IT": ["Network Security", "Cloud Computing", "IT Project Management"]
    }
    default_skills = ["Communication Skills", "Project Management", "Team Leadership"]
    return skills.get(department, default_skills)



# @app.route('/predict', methods=['GET', 'POST'])
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Extract form data
        data = {
            'Age': int(request.form['age']),
            'BusinessTravel': request.form['business_travel'],
            'DailyRate': int(request.form['daily_rate']),

            'DistanceFromHome': int(request.form['distance_from_home']),
            'Education': int(request.form['education']),
            'EducationField': request.form['education_field'],
            'EmployeeCount': 1,  # Assuming EmployeeCount is constant
            'EnvironmentSatisfaction': int(request.form['environment_satisfaction']),
            'Gender': request.form['gender'],
            'HourlyRate': int(request.form['hourly_rate']),
            'JobInvolvement': int(request.form['job_involvement']),
            'JobLevel': int(request.form['job_level']),
            'JobRole': request.form['job_role'],
            'JobSatisfaction': int(request.form['job_satisfaction']),
            'MaritalStatus': request.form['marital_status'],
            'MonthlyIncome': int(request.form['monthly_income']),
            'MonthlyRate': int(request.form['monthly_rate']),
            'NumCompaniesWorked': int(request.form['num_companies_worked']),
            'Over18': 1, 
            'OverTime': request.form['overtime'],
            'PercentSalaryHike': int(request.form['percent_salary_hike']),
            'PerformanceRating': int(request.form['performance_rating']),
            'RelationshipSatisfaction': int(request.form['relationship_satisfaction']),
            'StandardHours': 80,  # Assuming StandardHours is constant
            'StockOptionLevel': int(request.form['stock_option_level']),
            'TotalWorkingYears': int(request.form['total_working_years']),
            'TrainingTimesLastYear': int(request.form['training_times_last_year']),
            'WorkLifeBalance': int(request.form['work_life_balance']),
            'YearsAtCompany': int(request.form['years_at_company']),
            'YearsInCurrentRole': int(request.form['years_in_current_role']),
            'YearsSinceLastPromotion': int(request.form['years_since_last_promotion']),
            'YearsWithCurrManager': int(request.form['years_with_curr_manager'])
        }

        # Convert form data to DataFrame for prediction
        df = pd.DataFrame([data])

        # Encode categorical features
        for column in df.columns:
            if df[column].dtype == object and column in ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']:
                df[column] = le.fit_transform(df[column])
        
        
        # Make prediction
        prediction = xgb_model.predict(df)

        result = 'Laid off' if prediction[0] == 1 else 'Not laid off'
        skills = None
        if prediction[0] == 1:
            skills = recommend_skills(data['Department'], data['JobRole'])

        return render_template('predict_result.html', result=f'Employee is predicted to be: {result}.', skills=skills)

    return render_template('predict.html')


if __name__ == '__main__':
    app.run(debug=True)



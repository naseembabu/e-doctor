# E-Doctor  (Multiple Disease Predictor)
In this project, I trained machine learning models to predict the likelihood of Diabetes, Heart Disease, and Parkinson's Disease. The training was conducted using datasets from Kaggle, utilizing the Support Vector Machine (SVM) algorithm for the predictive modeling. To develop the graphical user interface (GUI), I employed the Streamlit library and deployed the application on the Streamlit cloud service.

The application's home page presents users with three disease prediction options, each corresponding to one of the targeted diseases. Users can select the disease they are concerned about and input relevant parameters. Based on this input, the system predicts whether the user is at risk of having the chosen disease. This intuitive interface ensures that users can easily navigate and utilize the prediction system to gain insights into their health conditions.

https://e-doctor-uu7shcobkcv9vgff4lpbxm.streamlit.app/

![desktop drawio](https://github.com/naseembabu/e-doctor/assets/71367662/634cb7e4-85ef-401a-bc63-67c5a9d8de7f)


![mobile drawio](https://github.com/naseembabu/e-doctor/assets/71367662/9ab58df7-1299-4d1e-8b77-b34b5450d6b7)



Certainly! Here’s a step-by-step guide in story form to help you create and deploy your "e-doctor" app using Streamlit and GitHub:

---

## Step-by-Step Guide to Deploying Your "e-doctor" App on Streamlit Cloud

### Step 1: Create a GitHub Repository
It all began with an idea to develop a revolutionary "e-doctor" app. You logged into your GitHub account and navigated to the "New Repository" section. You named your repository "e-doctor" and added a description to help others understand its purpose.

#### Steps:
1. Go to [GitHub](https://github.com).
2. Log in to your account.
3. Click on the "New" button to create a new repository.
4. Name the repository "e-doctor".
5. Add a description (optional but recommended).
6. Choose the repository visibility (public or private).
7. Click on "Create repository".

With the repository created, you then stored the necessary files, such as your Python scripts, Streamlit app script, and requirements.txt, into the "e-doctor" repository.

#### Steps:
1. Open your terminal or command prompt.
2. Navigate to the directory containing your project files.
3. Initialize Git and add your remote repository:
   ```bash
   git init
   git remote add origin https://github.com/naseembabu/e-doctor.git
   ```
4. Add your files and commit them:
   ```bash
   git add .
   git commit -m "Initial commit with all project files"
   ```
5. Push the files to GitHub:
   ```bash
   git push origin master
   ```

### Step 2: Sign Up on Streamlit Cloud
The next chapter in the story involved bringing the app to life on Streamlit Cloud. You visited [Streamlit Cloud](https://streamlit.io/cloud) and signed up.

#### Steps:
1. Go to [Streamlit Cloud](https://streamlit.io/cloud).
2. Click on the "Sign Up" button.
3. Choose to sign up using your GitHub account to streamline the process.

### Step 3: Connect Your GitHub Account
You connected your GitHub account with Streamlit Cloud to enable easy deployment of your repository.

#### Steps:
1. Authorize Streamlit Cloud to access your GitHub repositories.
2. Complete any additional setup steps required by Streamlit Cloud.

### Step 4: Deploy Your "e-doctor" App
With your accounts linked, it was time to deploy the app. You clicked on the "Create App" button to start the process.

#### Steps:
1. Click on the "Create App" button on Streamlit Cloud.
2. Select your "e-doctor" repository from the list of GitHub repositories.
3. Click on the "Deploy" button to start the deployment process.

### Step 5: Monitor and Finalize Deployment
You watched as Streamlit Cloud pulled your repository and started the deployment process. You kept an eye on the log files to ensure everything was running smoothly.

#### Steps:
1. Monitor the logs for any errors or issues.
2. Make sure that the requirements.txt file is correctly installed. If there are any missing dependencies, you may need to update your requirements.txt and redeploy.

### Step 6: Test Your "e-doctor" App
With the deployment complete, you finally got to test your "e-doctor" app in real-time. You input some values to see how the app performed and checked the model’s predictions.

#### Steps:
1. Open your deployed app on Streamlit Cloud.
2. Input some sample data to test the functionality.
3. Verify that the model is working as expected and providing accurate predictions.

### Conclusion
And just like that, your "e-doctor" app was live on Streamlit Cloud, ready to provide medical insights to users everywhere. You marveled at how the journey from code to a fully deployed app had been seamless and rewarding.

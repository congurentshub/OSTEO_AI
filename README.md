🖼 Sample Test Images Available

Sample medical images have been added to this repository inside the sample_images/ folder to help evaluators quickly test the model.

You can:

Download sample images from the repository

Upload them directly into the web application

View predictions and Grad-CAM visualizations instantly

This ensures smooth and easy evaluation without needing external data.

🔐 Authentication

To access the AI detection module, login is required.


🧪 Demo Credentials (For Evaluation)


Username: RADHE

Password: RADHE


🚀 Live Application

🔗 Live Demo: https://osteoai-bjnqfm4gqvusnjfibnzelg.streamlit.app/
🦴 Osteoporosis AI Detection System
📌 Overview

Osteoporosis AI is a deep learning–based web application designed to detect osteoporosis from medical images using advanced Convolutional Neural Networks (CNNs).

The system integrates:

🧠 AI-powered osteoporosis detection

🔥 Grad-CAM heatmap visualization for explainability

🤖 AI medical chatbot powered by Groq API

🔐 Secure login authentication

☁️ Live deployment via Streamlit Cloud

This project was developed as part of a healthcare AI hackathon to support early diagnosis and improve medical accessibility using Artificial Intelligence.

🚀 Live Application

🔗 Live Demo: https://osteoai-bjnqfm4gqvusnjfibnzelg.streamlit.app/

The application is deployed using Streamlit Cloud and integrated with GitHub for continuous deployment.






These credentials are provided strictly for demonstration purposes and do not grant access to any external systems.

In a production environment, authentication would be handled using encrypted credentials and secure role-based access control.

🧠 Model Details

Framework: PyTorch

Architecture: CNN-based deep learning model

Model Storage: Git LFS

Explainability: Grad-CAM Visualization

Deployment: Streamlit Cloud

The model predicts osteoporosis from uploaded medical images and highlights important regions using heatmap visualization to improve interpretability.

🤖 AI Medical Chatbot

The application includes an AI-powered chatbot to assist users with osteoporosis-related queries.

Powered by Groq API

Users can provide their own Groq API key at runtime

No API keys are stored in this repository

⚠ For security reasons, API keys are never published in the codebase.

📊 Key Features

✅ Real-time osteoporosis detection

🔥 Heatmap visualization using Grad-CAM

👤 Secure login system

🗄 Patient data handling (demo database)

🤖 AI-powered medical chatbot

☁️ Cloud deployment

🔒 Secure handling of secrets

🛠 Installation (Run Locally)

To run this project locally:

git clone https://github.com/congurentshub/OSTEO_AI.git
cd OSTEO_AI
pip install -r requirements.txt
streamlit run APPR.py


Make sure you are using Python 3.10 for compatibility.

📂 Project Structure
OSTEO_AI/
│
├── APPR.py                # Main Streamlit application
├── models/                # Trained model (stored via Git LFS)
├── assets/                # CSS and static files
├── database.db            # Demo database
├── requirements.txt       # Python dependencies
├── runtime.txt            # Python version configuration
└── .gitignore

🔒 Security & Best Practices

API keys are not stored in the repository

Git LFS is used for managing large model files

Authentication system implemented for controlled access

Deployment configured with secure environment handling

🏥 Problem Statement

Osteoporosis is often undiagnosed until fractures occur. Early detection is critical for preventive treatment and improved patient outcomes.

This AI-based system aims to:

Assist in early diagnosis

Support healthcare professionals

Increase accessibility in low-resource environments

Demonstrate the power of AI in medical imaging


📜 License

This project is developed for educational, research, and demonstration purposes.

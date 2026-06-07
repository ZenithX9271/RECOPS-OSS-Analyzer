# Setup Guide for RECOPS-OSS-Analyzer

## 1) Prerequisites  
- Install Miniconda (https://docs.conda.io/en/latest/miniconda.html)  
- Install Git (https://git-scm.com/) (needed for cloning the repository)  

---

## 2) Clone the repository  
Open a terminal (Command Prompt, PowerShell, or Anaconda Prompt) and run:  

```bash
git clone https://github.com/ZenithX9271/RECOPS-OSS-Analyzer.git
cd RECOPS-OSS-Analyzer
```

---

## 3) Create the environment  
The project uses a Conda environment specified in `environment.yml`.  
Run the following command to create it:  

```bash
conda env create -f environment.yml
```

---

## 4) Activate the environment  
Once the environment is created, activate it by running:  

```bash
conda activate recops_env
```

---

## 5) Run the Streamlit app  
Inside the project folder, launch the app with:  

```bash
streamlit run recops_app.py
```

This will open the app in your browser at:  
http://localhost:8501

---

## 6) Website  
You can now browse the web interface to explore the features that are available.  

---

## 7) Repository Analysis  
Paste the GitHub repository URL in the input box provided.  
After clicking outside the box, you’ll see a Start Analysis button.  
Clicking this will begin the feature extraction process for the repository.  

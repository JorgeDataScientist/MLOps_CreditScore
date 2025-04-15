from setuptools import setup, find_packages

setup(
    name="MLOps_CreditScore",
    version="0.3.0",
    description="MLOps pipeline to predict credit scores (Poor, Standard, Good) using a RandomForestClassifier with automated preprocessing, training, evaluation, deployment, and interactive visualization.",
    author="Jorge Luis Garcia",
    author_email="jorgeluisdatascientist@gmail.com",
    packages=find_packages(),
    install_requires=[
        "hydra-core>=1.3.2",
        "bentoml>=1.4.7",
        "scikit-learn>=1.6.1",
        "streamlit>=1.0.0",
        "mlflow>=2.21.3",
        "dvc>=3.59.1",
        "pandas>=2.2.3",
        "numpy>=2.2.4",
        "pytest>=8.3.5",
        "omegaconf>=2.3.0",
    ],
    python_requires=">=3.8",
)
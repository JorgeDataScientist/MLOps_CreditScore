from setuptools import setup, find_packages

setup(
    name="intelligent_credit_scoring",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas==2.2.2",
        "numpy==1.26.4",
        "scikit-learn==1.5.1",
        "hydra-core==1.3.2",
        "mlflow==2.15.1",
        "dagshub==0.3.34",
        "dvc==3.53.2",
        "pandera==0.20.3",
        "joblib==1.4.2",
        "matplotlib==3.9.2",
        "seaborn==0.13.2",
        "bentoml==1.3.2",
        "python-dotenv==1.0.1",
    ],
)
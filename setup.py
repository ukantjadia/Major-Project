import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.0"


REPO_NAME = "Heart-Disease-Classification-with-Electrocardiogram"
AUTHOR_USER_NAME = "UkantJadia"
SRC_REPO = "heartDiseaseClassification"
AUTHORE_EMAIL = "ukantjadia0120@gmail.ocom"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHORE_EMAIL,
    description="A small project to demonstarte MLOPS",
    long_description=long_description,
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
)

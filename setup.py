from setuptools import find_packages,setup

setup(
    name='mcqgenrator',
    version='0.0.1',
    author='Nimisha Tiwari',
    author_email='tiwarinimisha206@gmail.com',
    install_requires=["openai","langchain","flask","python-dotenv","PyPDF2"],
    packages=find_packages()
)
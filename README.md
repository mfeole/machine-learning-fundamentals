# Machine Learning Fundamentals

## Introduction

This project aims to describe machine learning fundamentals from an introductory mathematical side, to the models, methods, and algorithms that implement those models. Each subfolder will consist of Jupyter notebooks with an explanation of the math background and of the essential algorithms, and also executable (not restricted to-)Python scripts that apply those algorithms to particular examples.

Also other data science concepts, methods and tools related to machine learning will also be provided.

## Machine Learning basics

Machine learning is the generalization of the classic regression problem in statistics. It is the task of estimating a function <img src="https://latex.codecogs.com/svg.latex?\Large&space;f(x;\theta)" title="\Large f(x;\theta)" /> for <img src="https://latex.codecogs.com/svg.latex?\Large&space;f(x)" title="\Large f(x)" />, such that

<img src="https://latex.codecogs.com/svg.latex?\Large&space;Y=f(X)+\epsilon" title="\Large Y=f(X)+\epsilon" />

given observations <img src="https://latex.codecogs.com/svg.latex?\Large&space;(x^i,y^i)_{i=1}^N" title="\Large (x^i,y^i)_{i=1}^N" /> of <img src="https://latex.codecogs.com/svg.latex?\Large&space;(X,Y)" title="\Large (X,Y)" /> (in the supervised case), where the noise term <img src="https://latex.codecogs.com/svg.latex?\Large&space;\epsilon" title="\Large \epsilon" /> is assumed to be such that <img src="https://latex.codecogs.com/svg.latex?\Large&space;E[\epsilon]=0" title="\Large E[\epsilon]=0" />. The task is then to find a good estimator of the model parameters: <img src="https://latex.codecogs.com/svg.latex?\Large&space;\theta" title="\Large \theta" />.

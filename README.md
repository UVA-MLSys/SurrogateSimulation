# Surrogate Simulation for Earthquake Prediction

#### Vishal Kamalakrishnan, Zach Russell, Kevin Ma, and Justin Zheng
#### Research Mentors: Prof. Geoffrey Fox, Prof. Judy Fox, Md. Khairul Islam
#### UVA-MLSYS Summer Research Program 2023

## Project Overview
Our mission is to develop earthquake prediction model that combines the accuracy and efficiency. Earthquakes, with their sudden and devastating impact, call for timely and reliable solutions to minimize their consequences on society and infrastructure. Our goal is to create a surrogate model that matches the precision of traditional physics-based models while drastically reducing computation time

We compared many iterative models including XGB Regressor, Support Vector Regression, MLP Regressor, Linear Regression, and Random Forest. We evaluated each model's strengths and weaknesses including their accuracy and feature importance. Based on our research we were able to recommend a hypothetical model for VAE employing generative and iterative solutions to better model earthquakes, as well as suggest several other options that could be feasibly be developed in the near future

## Resources
We utilized Prof. John Rundle's ETAS simulation code to build a surrogate model out of. 
- [How to run ETAS](./ETAS%20Codes%20Updated%205-16-2023/ETAS/instructions.md)
- Citations and links can be found in our Github Pages https://uva-mlsys.github.io/SurrogateSimulation/data_visualization.html

## Repository Contents
- Formatted_ETAS_Output.csv : contains formatted output from ETAS simulation code which serves as the input data for the surrgoate
- data_visualization.ipynb : contains input data and charts off of ETAS

# Hypothetical Model for VAE

VAE Architecture:
Encoder: a feedforward neural network that will take earthquake data from the ETAS model as input to a lower-dimensional latent space representation. As this is hypothetical, we would need to figure out what components of the encoder architecture that would need to be implemented, however we imagine that it will contain several fully connected layers followed by activation functions (Rectified Linear Unit). The last layer will produce the mean and variance of the latent distribution (Gaussian Distribution)

Decoder: a feedforward neural network that takes the latent representation produced by the encoder and reconstructs the earthquake data. It would also contain fully connected layers with activation functions. THe output layer of the decoder will generate the reconstruction earthquake data.

Reconstruction loss: Measures how well the decoder can reconstruct the original earthquake data from the latent space. We would most likely use mean squared error, but other choices we could use to measure this would be binary cross-Entropy (BCE) loss, depending on the earthquake data.

Kullback-Leibler (KL) divergence: Measures the difference between the learned latent distribution versus the desired distribution. Acts as a regularization term, encouraging the learned latent space to be structured and continuous

Total loss used for training would be a combination of the reconstruction loss and the KL divergence, weighted by hyperparameters that control the importance of each term.

Training the VAE:
We would then train the VAE using the earthquake dataset produced by the ETAS model. During this training, the encoder will map the earthquake data into the latent space, and the decoder will attempt to reconstruct the original data from the latent space. The VAE’s parameters (weights and biases) are updated to minimize the combined loss using optimization techniques like Stochastic Gradient Descent (SGD) or Adam.

Latent space exploration:
Once the VAE is trained, you can explore the latent space to understand the learned representations better. Visualize the latent space using dimensionality reduction techniques (e.g., t-SNE or PCA) to see if the earthquake sequences cluster in meaningful ways. By 

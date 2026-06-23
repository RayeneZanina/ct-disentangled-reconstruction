# CT brain image reconstruction and pathology extraction

This project presents a model for reconstructing CT images using a physics-based model with learned priors. The image is decomposed into a healthy anatomy and a residual that catches pathological structures that disagree with the healthy prior.

The idea of this approach is to use a convolutional Adversarial AutoEncoder (AAE), not as a tool for generating images, but as a model that can learn healthy anatomical priors. The reconstructed image is then constrained to be in the healthy manifold learned by the AAE, while the pathologies caught by the measurements form a residual image that isolates the tumor. This is a way to both reconstruct images in better quality under low-dose CT, but also automically extract the tumor without any post-processing or segmentation. 

The AAE learns a compact representation of the image $z = E(x)$, and the decoder reconstructs healthy anatomy $x_{healthy} = D(z)$. Not only does this allow the AAE to learn the manifold of healthy images, but also it reduces the dimensionality of the optimization problem. To further reduce the dimensionality, PCA is performed on the latent vectors, which restricts the reconstructed image to have healthy anatomy. The latent vector is then represented as $z = \mu + V^T c$. Using these tools, we can represent the image as

$$x = x_{healthy} + r = D(\mu + V^T c) + r$$


The imaging system is modeled using the Radon transform $y = Ax$. The photon counts are modeled using a Poisson distribution

$$y_i \sim Poisson(I_0 e^{(Ax)_i})$$

We then use Maximum Likelihood Estimation to find $x$. We try to maximize $p(y|x)$. From the Poisson model, we know that $p(y_i | x) = \frac{(I_0 e^{(Ax)_i})^{y_i} e^{-I_0 e^{(Ax)_i}}}{y_i!}$. We then assume that all bins are independent, giving us $p(y|x) = \prod_i p(y_i|x)$. For this optimization problem, we try to minimize the log-likelihood

$$-\log(p(y|x)) = -\sum_i y_i \log(I_0 e^{-(Ax)_i}) - I_0 e^{-(Ax)_i} - \log(y_i!) = \sum_i y_i (Ax)_i + I_0 e^{-(Ax)_i} - y_i \log(I_0) + \log(y_i!)$$

Our goal is then to find $\hat{x} = \arg \min -\log(p(y|x))$, which we solve using gradient-based methods. 

We note that $- y_i \log(I_0) + \log(y_i!)$ does not depend on $x$, meaning they are constants when taking the gradient of $x$, and can be removed. Something else that is noteworthy is that our current objective function $L(x) = \sum_i y_i (Ax)_ i + I_0 e^{-(Ax)_ i}$ is equivalent to the generalized Kullback-Leibler divergence $D_{KL}(y||x) = \sum_i y_i \log(\frac{y_i}{I_0 e^{-(Ax)_i}}) - y_i + I_0 e^{-(Ax)_i}$, up to a constant, meaning their gradients are identical. However, the KL divergence is always positive, leading to a more insightful and intuitive objective function.

By combining this objective function and our decomposition of the image $x$, we get our final objective function

$$L(c, r) = \sum_i y_i (A D(\mu + V^T c) + r)_i + I_0 e^{-(A D(\mu + V^T c) + r)_i} + R(c, r)$$ 

where $R$ is the regularization.

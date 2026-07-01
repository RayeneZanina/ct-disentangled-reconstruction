# CT brain image reconstruction and pathology extraction

This project presents a model for reconstructing CT images using a physics-based model with learned priors. The image is decomposed into a healthy anatomy and a residual that catches pathological structures that disagree with the healthy prior.


For this project, we use images from BrainWeb from the McConnell Brain Imaging Centre (http://www.bic.mni.mcgill.ca/brainweb/). These are high-quality MRI phantoms with the different tissues seperated, which allows us to use attenuation coefficients on the different tissues to reconstruct a CT phantom. Unfortunately, I don't have access to a large amount of phantoms, so I used a few transforms (rotation, scaling, stretching) and made the attenuation coefficients normally distributed to generate a larger dataset. It is still far from ideal. The tumors are also synthetically added using gaussian blobs, which is also a very simple model, but is just serving as an example.


The idea of this approach is to use a convolutional Adversarial AutoEncoder (AAE), not as a tool for generating images, but as a model that can learn healthy anatomical priors. The reconstructed image is then constrained to be in the healthy manifold learned by the AAE, while the pathologies caught by the measurements form a residual image that isolates the tumor. This is a way to both reconstruct images in better quality under low-dose CT, but also automically extract the tumor without any post-processing or segmentation. 

The AAE learns a compact representation of the image $z = E(x)$, and the decoder reconstructs healthy anatomy $x_{healthy} = D(z)$. Not only does this allow the AAE to learn the manifold of healthy images, but also it reduces the dimensionality of the optimization problem. The discriminator also imposes a prior on the manifold of healthy images on the latent space to resemble a normal distribution. This allows a smoother latent space and facilitates the optimization problem. This is why we prefer AAE over a GAN although it can produce images with lower quality as GANs are not tailored for inverse problems. We also prefer AAE over VAE as the adversarial loss is less constraining than the KL-divergence loss, which resulted is sharper images when tested.

The VAE images are much blurrier as seen here.


<img width="552" height="291" alt="image" src="https://github.com/user-attachments/assets/cff17c69-74ad-454d-a966-ce28b003c403" />


The AAE can produced sharper images.


<img width="552" height="291" alt="image" src="https://github.com/user-attachments/assets/adccdc5d-5c01-4ce5-8888-f5a7172fd5e9" />



To further reduce the dimensionality, PCA is performed on the latent vectors, which restricts the reconstructed image to have healthy anatomy. The latent vector is then represented as $z = \mu + V^T c$. Using these tools, we can represent the image as

$$x = x_{healthy} + r = D(\mu + V^T c) + r$$


To determine how many components to use, we plot the cumulative explained variance as a function of the number of components. We can see that 40 components can capture around 95% of the variance.


<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/814761ea-c029-42fb-9bdf-12c0311c9191" />


The image can be recovered very well from the PCA components.


<img width="831" height="418" alt="image" src="https://github.com/user-attachments/assets/eac8db93-b7ad-4776-9f6a-73ef7d1a37d6" />


The imaging system is modeled using the Radon transform $y = Ax$. The photon counts are modeled using a Poisson distribution

$$y_i \sim Poisson(I_0 e^{(Ax)_i})$$

In the simulated sinograms $y$, we also implement a gaussian noise for the electronic noise. It isn't taken into account into the model for the optimization problem though.

We then use Maximum Likelihood Estimation to find $x$. We try to maximize $p(y|x)$. From the Poisson model, we know that $p(y_i | x) = \frac{(I_0 e^{(Ax)_i})^{y_i} e^{-I_0 e^{(Ax)_i}}}{y_i!}$. We then assume that all bins are independent, giving us $p(y|x) = \prod_i p(y_i|x)$. For this optimization problem, we try to minimize the negative log-likelihood

$$-\log(p(y|x)) = -\sum_i y_i \log(I_0 e^{-(Ax)_i}) - I_0 e^{-(Ax)_i} - \log(y_i!) = \sum_i y_i (Ax)_i + I_0 e^{-(Ax)_i} - y_i \log(I_0) + \log(y_i!)$$

Our goal is then to find $\hat{x} = \arg \min -\log(p(y|x))$, which we solve using gradient-based methods. 

We note that $- y_i \log(I_0) + \log(y_i!)$ does not depend on $x$, meaning they are constants when taking the gradient of $x$, and can be removed. Something else that is noteworthy is that our current objective function $L(x) = \sum_i y_i (Ax)_ i + I_0 e^{-(Ax)_ i}$ is equivalent to the generalized Kullback-Leibler divergence $D_{KL}(y \lVert x) = \sum_i y_i \log(\frac{y_i}{I_0 e^{-(Ax)_i}}) - y_i + I_0 e^{-(Ax)_i}$, up to a constant, meaning their gradients are identical. However, the KL divergence is always positive, leading to a more insightful and intuitive objective function.

By combining this objective function and our decomposition of the image $x$, we get our final objective function

$$L(c, r) = \sum_i( y_i \log(\frac{y_i}{I_0 e^{-(AD(\mu + V^T c) + r)_i}}) - y_i + I_0 e^{-(AD(\mu + V^T c) + r)_i}) + R(c, r)$$ 

where $R$ is the regularization. There are multiple choices of regularization functions, and it is possible to also train a model to learn a prior on our variables and use it as a regularization term. This learned prior can replace the regularization. However, I went with a pretty simple choice. 

$$ R(c, r) = \lambda_c \lVert{c} \rVert^2_2 + \lambda_r \lVert{r} \rVert_1 + \lambda_n \lVert{\Delta r} \rVert^2_2 $$

The L2 regularization punishes deviations of the PCA coefficients from the healthy anatomical manifold, the L1 regularization on $r$ encourages sparsity and the laplacian avoids $r$ catching the noise. The final objective function is then

$$L(c, r) = D_{KL}(y \lVert x) + \lambda_c \lVert{c} \rVert^2_2 + \lambda_r \lVert{r} \rVert_1 + \lambda_n \lVert{\Delta r} \rVert^2_2$$

The gradient of this objective function is then used to update $\hat{c}$ and $\hat{r}$. We initialize $\hat{c}_ 0 = PCA(E(x_{FBP}))$ and $\hat{r}_0 = 0$. The FBP image serves as a good initalization, and prevents the residual from taking over the healthy brain anatomy. Although we regularize $r$ to prevent it from learning the noise, we still expect some leakage of the noise into the residual. We can process $r$ to remove signals below a certain threshold to get a cleaner image. 

We try this for a sample phantom at high dose ($I_0 = 5 \cdot 10^3$).


<img width="1606" height="403" alt="image" src="https://github.com/user-attachments/assets/4a66968c-f4a9-4758-9ac1-79e501a5928a" />


We compare this to the filtered backprojected image.


<img width="1218" height="407" alt="image" src="https://github.com/user-attachments/assets/b9bea1ef-91ff-419d-8d9f-570b70472807" />


We then try this for a lower dose ($I_0 = 10^3$).

<img width="1606" height="384" alt="image" src="https://github.com/user-attachments/assets/48ff3817-e29a-4db7-9e69-dc43f1019924" />

We compare this to the FBP image again.

<img width="1218" height="407" alt="image" src="https://github.com/user-attachments/assets/ce26eea4-f922-4ce5-a974-30386f44837e" />


However, we do get lesser quality reconstructions for phantoms from a different source. This is very likely due to the small dataset that I started with, which led to the AAE not being able to learn the full manifold of healthy anatomical brain images, which also leads to the residual not acting properly.


<img width="1606" height="403" alt="image" src="https://github.com/user-attachments/assets/8689e467-7a80-4118-a8d7-7a489fe99ce3" />



We can compute the PSNR and SSIM as metrics to compare the performance of our model to FBP with hamming filter. For the phantom from the different source.

<img width="663" height="459" alt="image" src="https://github.com/user-attachments/assets/128db331-925c-411b-bf7b-ba7ca3a4e698" />

<img width="658" height="459" alt="image" src="https://github.com/user-attachments/assets/78418ac3-e8b0-4f12-8b3f-6d3abfeeb58a" />

The model outperforms the FBP for low dose, but for a higher dose it starts fitting to the training set too much. We can confirm this by using samples from the training set that are slightly modified when comparing. 

<img width="663" height="459" alt="image" src="https://github.com/user-attachments/assets/7071a75a-d2ce-4c55-a30e-45363b38e7e6" />

<img width="658" height="459" alt="image" src="https://github.com/user-attachments/assets/08e60084-96d0-4ca3-9a34-0f437de28e4d" />

With a larger dataset, this model could theoretically learn the manifold of healthy images better, but it is very restricted right now due to the small dataset. Another point I want to mention is that, theoretically the AAE can also generate a lot more images which can be used to train an unrolled network to learn the optimization steps, which would make the whole optimization process a lot faster.















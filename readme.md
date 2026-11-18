# i `like_streams`.

Here we'll compare perturbed particle spray streams to a "1/x" filter shape, mimicking the characteristic impact shape from, e.g., Yoon+2011, where the kick toward the impact site is given by

$$
\Delta v_x = -\frac{2 G M_{\rm sub}v_{\perp}^2 x}{v_{\rm enc}b^2(v_{\perp}^2(x/b)^2 + v_{\rm enc}^2)}\\
 \\
= -\frac{2 G M_{\rm sub}}{v_{\rm enc}} \frac{x}{x^2 + (\frac{b v_{\rm enc}}{v_{\perp}})^2}
$$

We'll model the shapes of signals then as 
$$
y = \frac{ax}{x^2 + w^2},
$$
where $a = -\frac{2 G M_{\rm sub}}{v_{\rm enc}}$ and $w=\frac{b v_{\rm enc}}{v_{\perp}}$. The extrema of this expression are located at $\pm w$ (the "width" of the signal) and have values $A=\pm a/2w$ (intuitively this makes sense, by making $w$ larger we're decreasing the perpendicular relative to the total encounter velocity, both broadening the perturbation and reducing amplitude). We can also write

$$
y = \frac{(2wA)(x-l)}{(x-l)^2 + w^2}, \tag{1} = A\left[\frac{2w(x-l)}{(x-l)^2 + w^2}\right]
$$

where now $A$ is the actual amplitude of the signal, and $l$ is its offset from zero so that the extrema are at $l\pm w$.

If the stream has a tilt and is not centered exactly at $y=0$ it becomes convenient to add a linear term, so that
$$
y = A\left[\frac{2w(x-l)}{(x-l)^2 + w^2}\right] + [m(x-l)+b].
$$

The model is linear in $A, m, b$ and nonlinear in $w, l$. We'll use a Gaussian likelihood to optimize for the best parameters, requiring that in each phase space dimension we add the intrinsic stream width parameter $\sigma_{\rm stream}$. 

With some algebra (can add here), we can show that for any given $(w, l, \sigma_{\rm stream})$ there exists a closed form expression for the highest-likelihood $A, m, b$ since the model is linear in those parameters (this speeds up the optimization considerably). 
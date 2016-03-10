<h3 style="text-align: center;">http://x.x.x.x:8000</h3>



<h2 style="margin-top:15%">Outline</h2>

#### Classification

#### Discriminative Approach

#### Generative Approach

#### Bayesian Inference in Stan

#### Demo

<h4 style="text-align: center; margin-top:15%">http://x.x.x.x:8000</h4>



<section data-background="assets/three_classes_1.png">
  <h3>The Classification Problem</h3>
  <h4>Have</h4>
  <p>$D$-dimensional, real explanatory variable $\\boldsymbol{X} = \\left(X\_1, \\ldots, X\_D\\right)^{\\intercal}$</p>
  <p>Class label $C$, assumes one out of $K$ classes $\\left\\{1, \\ldots, K\\right\\}$</p>
  <p>Training set of $N$ observations $\\left(\\boldsymbol{x}, c\\right)\_{1:N}$</p>
  <h4>Want</h4>
  <p>Assign each new observation $\\boldsymbol{x}'\_{n'}$, $n'=1,\\ldots,N'$, to *one* class</p>
  <p>
    \\[
      \\boldsymbol{x}'\_{n'} \\rightarrow \\boldsymbol{x}'\_{n'}, c'\_{n'}
    \\]
  </p>
  <p>How can we infer the labels $c'\_{1:N'}$?</p>
</section>



<section data-background="assets/no_touching.png">
  <h2>Discriminative Approach</h2>
  <p>Train a model so as to maximize the probability of getting correct labels</p>
</section>



<section data-background="assets/workhorse.png">
  <h3>Logistic "Regression"</h3>
  <p>Logistic regression estimates $p\\left(c=k \\, \\middle| \\, \\boldsymbol{x}\\right)$, $k=1,\\ldots,K$</p>
  <p>Idea: Map the output of a linear model, $b + \\boldsymbol{w}^{\\intercal} \\boldsymbol{x}$, to a probability</p>
  <p>
    \\[
      p\\left(c=k \\, \\middle| \\, \\boldsymbol{x}, \\boldsymbol{b}, \\boldsymbol{W}\\right) = \\operatorname{softmax}\_k(\\boldsymbol{b} + \\boldsymbol{W} \\boldsymbol{x}) = \\frac{\\exp\\left(b\_k + \\boldsymbol{w}\_k^{\\intercal} \\boldsymbol{x}\\right)}{\\sum\_{k'=1}^K \\exp\\left(b\_{k'} + \\boldsymbol{w}\_{k'}^{\\intercal} \\boldsymbol{x}\\right)}
    \\]
  </p>
  <p>Softmax is a soft activation function with probability interpretation:</p>
  <p style="text-align: center">
    <img src="/assets/softmax.svg" style="margin: none; background: none; border: none; box-shadow: none;">
    <br/>
    Graph of $y = \\left(1+\\exp(-x)\\right)^{-1}$
  </p>
</section>



<section data-background="assets/pumping_iron.png">
  <h3>Training<h3>
  <h4>Maximum Likelihood Estimation</h4>
  <p>Maximize the conditional log-likelihood of the training data $\\left(\\boldsymbol{x}, c\\right)\_{1:N}$:</p>
  <p>
    \\[
      \\operatorname{argmax}\_{\\boldsymbol{b}, \\boldsymbol{W}} \\sum\_{n=1}^N \\log p\\left(c\_n \\, \\middle| \\, \\boldsymbol{x}\_n, \\boldsymbol{b}, \\boldsymbol{W}\\right)
    \\]
  </p>
  <h4>Maximum A Posteriori Estimation</h4>
  <p>Add $\\ell\_2$-regularization and maximize:</p>
  <p>
    \\begin{multline}
      \\operatorname{argmax}\_{\\boldsymbol{b}, \\boldsymbol{W}} \\sum\_{n=1}^N \\log p\\left(c\_n \\, \\middle| \\, \\boldsymbol{x}\_n, \\boldsymbol{b}, \\boldsymbol{W}\\right) - \\lambda \\sum\_{k=1}^K \\left(\\left|b\_k\\right|^2 + \\lVert \\boldsymbol{w}\_k \\rVert^2\\right) = \\\\
        \\operatorname{argmax}\_{\\boldsymbol{b}, \\boldsymbol{W}} \\prod\_{n=1}^N p\\left(c\_n \\, \\middle| \\, \\boldsymbol{x}\_n, \\boldsymbol{b}, \\boldsymbol{W}\\right) \\times \\prod\_{k=1}^K \\left[\\operatorname{Normal} \\left(b\_k \\middle| 0, \\tfrac{1}{2 \\lambda}\\right) \\operatorname{Normal} \\left(\\boldsymbol{w}\_k \\middle| \\boldsymbol{0}, \\tfrac{\\boldsymbol{1}}{2 \\lambda}\\right)\\right]
    \\end{multline}
  </p>
</h3>



<section data-background="assets/three_classes_2.png">
  <h3>Prediction</h3>
  <p>Discrimination is based on hard boundaries:</p>
  <p>
    \\begin{align}
      c'\_{n'} & = \\operatorname{argmax}\_k p\\left(c'\_{n'}=k \\, \\middle| \\, \\boldsymbol{x}'\_{n'}, \\boldsymbol{b}, \\boldsymbol{W}\\right) \\\\
        & = \\operatorname{argmax}\_k \\operatorname{softmax}\_k\\left(\\boldsymbol{b} + \\boldsymbol{W} \\boldsymbol{x}'\_{n'}\\right)
    \\end{align}
  </p>
</section>



<section data-background="assets/the_thinker.png">
  <h3>Problems</h3>
  <p>How to choose the regularization parameter $\\lambda$ so as to avoid overfitting?</p>
  <p>How to incorporate prior knowledge?</p>
  <p>How to integrate this method into a larger model?</p>
  <p>Isn't discrimination hiding the fact that there are uncertainties in the prediction?</p>
</section>



<section data-background="assets/casino_royal.png">
  <h2>Bayesian Generative Approach</h2>
  <p>Build a full probabilistic model of all variables, not only the class label</p>
</section>



<section data-background="assets/sherlock.png">
  <h3>Bayesian Learning</h3>
  <p>
    <ol>
      <li>
        <p>Introduce a *generative model* of the data, $\\boldsymbol{x}\_{1:N}$, conditioned on the class labels $c\_{1:N}$ and other latent variables $\\theta$:</p>
        <p>
          \\[
            p\\left(\\boldsymbol{x}\_{1:N} \\middle| c\_{1:N}, \\theta\\right)
          \\]
        </p>
      </li>
      <li>
        <p>Model a *prior probability* for the latent variables:</p>
        <p>
          \\[
            p\\left(c\_{1:N}, \\theta\\right)
          \\]
        </p>
      </li>
      <li>
        <p>Together, that will give us the *posterior probability*:</p>
        <p>
          \\[
            p\\left(c\_{1:N}, \\theta \\middle| \\boldsymbol{x}\_{1:N}\\right) \\propto p\\left(\\boldsymbol{x}\_{1:N} \\middle| c\_{1:N}, \\theta\\right) p(c\_{1:N}, \\theta)
          \\]
        </p>
        <p>(Bayes theorem)</p>
      </li>
    </ol>
  </p>
</section>



<section data-background="assets/price.png">
  <h3>Why Bayesian Learning?</h3>
  <p>Resistant to noise, avoids overfitting</p>
  <p>Takes into account prior knowledge, better results in smaller samples</p>
  <p>More flexibility, straightforward integration into larger model</p>
  <p>Predictions are probabilistic rather than discriminative</p>
</section>



### A General Generative Model for Classification

We imagine that

\\[
  \\left. \\boldsymbol{x}\_n \\, \\middle| \\, c\_n, \\phi\_{1:K} \\right. \\sim f\\left(\\phi\_{c\_n}\\right)
\\]

In words: *Given the label $c\_n$, each observed data point $\\boldsymbol{x}\_n$ is drawn from some distribution with probability density $f\\!\\left(\\phi\_{c\_n}\\right)$, where the $\\phi\_{k}$ are latent parameters*

<img src="/assets/hierarchical_model_1.svg" style="margin: none; background: none; border: none; box-shadow: none;">



<section data-background="assets/propeller_hat.png">
  <h3>A Naïve Assumption</h3>
  <p>As a simplification, assume that the features factorize:</p>
  <p>
    \\[
      p\\left(\\boldsymbol{x}\_n \\middle| c\_n, \\phi\_{1:K}\\right) = \\prod\_{i=1}^{D} \\, f\\left(x\_{ni} \\middle| \\phi\_{c\_n i}\\right)
    \\]
  </p>
  <p>
    <img src="/assets/hierarchical_model_2.svg" style="margin: none; background: none; border: none; box-shadow: none;">
  </p>
</section>



### Generate Normally Distributed Samples

For reasons which will become clear soon, choose a normal distribution:

\\begin{align}
  f\\left(x\_i \\middle| \\phi\_{ki}\\right) & \\equiv \\operatorname{Normal}\\left(x\_i \\middle| \\mu\_{ki}, \\sigma\_i^2\\right) \\\\
  & = \\frac{1}{\\sigma\_i \\sqrt{2 \\pi}} \\, \\exp\\left[- \\frac{x\_i - \\mu\_{ki}}{\\sigma\_i}\\right]
\\end{align}

with latent means $\\mu\_{ki}$ and variances $\\sigma\_{i}^2$

<img src="/assets/hierarchical_model_3.svg" style="margin: none; background: none; border: none; box-shadow: none;">



### A Class Indicator Prior

We further imagine that

\\[
  \\left. c\_n \\, \\middle| \\, \\pi\_{1:K} \\right. \\sim \\operatorname{Categorical}\\left(\\pi\_{1:K}\\right)
\\]

In words: *The indicator variables are distributed according to a categorical distribution defined on event probabilities $\pi\_{k}$ with $0 \\le \\pi\_{k} \\le 1$ and $\\sum\_{\\! k=1}^{\\! K} \\pi\_{k} = 1$*

<img src="/assets/hierarchical_model_4.svg" style="margin: none; background: none; border: none; box-shadow: none;">


### The Categorical Distribution

#### Probability Mass Function

Loaded $K$-sided die roll:

\\[
  p\\left(c\_n=k \\middle| \\pi\_{1:K}\\right) = \\pi\_{k}
\\]

<p style="text-align: center; margin-top: 10%">
  <img src="/assets/loaded_die.gif" style="margin: none; background: none; border: none; box-shadow: none;">
</p>



<section data-background="assets/tamed_zebras.png">
  <h3>The Connection to Logistic Regression</h3>
  <p>With a naïve Gaussian prior on $\\boldsymbol{X}$ and a categorical prior on $C$, we have:</p>
  <p>
    \\begin{align}
      p\\left(c\_n=k \\, \\middle| \\, \\boldsymbol{x}\_n, \\pi\_{1:K}, \\boldsymbol{\\mu}\_{1:K}, \\boldsymbol{\\sigma}\\right) & = \\frac{\\pi\_k \\; \\operatorname{Normal}\\left(\\boldsymbol{x}\_n \\middle| \\, \\boldsymbol{\\mu}\_k, \\operatorname{diag}\\left(\\boldsymbol{\\sigma}^2\\right)\\right)}{\\prod\_{k'=1}^K \\pi\_{k'} \\; \\operatorname{Normal}\\left(\\boldsymbol{x}\_n \\middle| \\, \\boldsymbol{\\mu}\_{k'}, \\operatorname{diag}\\left(\\boldsymbol{\\sigma}^2\\right)\\right)} \\\\
      & \\; \\vdots \\\\
      & = \\operatorname{softmax}\_k(\\boldsymbol{b} + \\boldsymbol{W} \\boldsymbol{x}\_n)
    \\end{align}
  </p>
  <p>with</p>
  <p>
    \\begin{align}
      b\_k & = \\textstyle \\log \\pi\_k - \\sum\_{i=1}^D \\frac{\\mu\_{ki}^2}{2 \\sigma\_i^2} \\\\
      w\_{ki} & = \\textstyle \\frac{\\mu\_{ki}}{\\sigma\_i^2}
    \\end{align}
  </p> 
  <p>That's the well-known likelihood function of logistic regression!</p>
</section>



### A Class Prevalence Prior

We choose a *conjugate prior* for the probabilities:

\\[
  \\left. \\pi\_{1:K} \\, \\middle| \\, \\alpha \\right. \\sim \\operatorname{Dirichlet}\\left(\\alpha\\right)
\\]

In words: *The probabilities $\pi\_{k}$ are distributed according to a symmetric Dirichlet distribution with concentration parameter $\\alpha > 0$*

<img src="/assets/hierarchical_model_5.svg" style="margin: none; background: none; border: none; box-shadow: none;">


### The Dirichlet Distribution I

#### Probability Density Function

For the symmetric Dirichlet distribution:

\\[
  p\\left(\\pi\_{1:K} \\middle| \\alpha\\right) = \\frac{\\Gamma\\left(K \\alpha\\right)}{\\Gamma\\left(\\alpha\\right)^{K}} \\prod\_{k=1}^{K} \\pi\_k^{\\alpha - 1}
\\]

<div>
  <div style="float: left; width: 33.33%; text-align: center;">
    <img src="/assets/dirichlet_small.svg" style="margin: none; background: none; border: none; box-shadow: none;">
    <br/>
    If $\\alpha \\ll 1$, then concentrated around the corners of the simplex
  </div>
  <div style="display: inline-block; width: 33.33%; text-align: center;">
    <img src="/assets/dirichlet_one.svg" style="margin: none; background: none; border: none; box-shadow: none;">
    <br/>
    If $\\alpha = 1$, then uniformly distributed over the simplex
  </div>
  <div style="float: right; width: 33.33%; text-align: center;">
    <img src="/assets/dirichlet_large.svg" style="margin: none; background: none; border: none; box-shadow: none;"><br/>
    If $\\alpha \\gg 1$, then concentrated around the center of the simplex
  </div>
</div>


### The Dirichlet Distribution II

#### Pólya's urn

Consider an urn containing balls of $K$ different colors.

Initially, the urn contains each $\\alpha$ balls of colors $1$, $2$, $\\ldots$, $K$.

Now perform $N$ draws from the urn, where after each draw, the ball is placed back into the urn with an additional ball of the same color.

In the limit as $N \\to \\infty$, the proportions $\\pi\_{1:K}$ of different colored balls in the urn will be distributed as $\\operatorname{Dirichlet}(\\alpha)$.



### Base Measures

Pick distributions for the $\\mu\_{ki}$ and $\\sigma\_{i}$:

\\begin{align}
  \\left. \\mu\_{ki} \\, \\middle| \\, \\mu\_0, \\kappa\_0 \\right. & \\sim \\operatorname{Normal}\\left(\\mu\_0, \\sigma\_{i}^{2} / \\kappa\_0\\right) \\\\
  \\left. \\sigma\_{i}^{-2} \\middle| \\, a\_0, b\_0 \\right. & \\sim \\operatorname{Gamma}\\left(a\_0, b\_0\\right)
\\end{align}

with prior mean $\\mu\_0$, prior sample size $\\kappa\_0$, shape $a\_0$, and rate $b\_0$

<img src="/assets/hierarchical_model_6.svg" style="margin: none; background: none; border: none; box-shadow: none;">



### The Posterior of The Complete Model

\\begin{multline}
  p\\left(c\_{1:N}, \\pi\_{1:K}, \\boldsymbol{\\mu}\_{1:K}, \\boldsymbol{\\sigma} \\, \\middle| \\, \\boldsymbol{x}\_{1:N}, \\alpha, \\mu\_0, \\kappa\_0, a\_0, b\_0\\right)
    \\\\ \\propto \prod\_{n=1}^{N} \\left[\\prod\_{i=1}^{D} p\\left(x\_{ni} \\middle| \\mu\_{c\_n i}, \\sigma\_i^2\\right)\\right] p\\left(c\_n \\middle| \\pi\_{1:K}\\right) \\quad
    \\\\ \\times p\\left(\\pi\_{1:K} \\middle| \\alpha\\right) \\prod\_{i=1}^{D} \\left[\\prod\_{k=1}^{K} p\\left(\\mu\_{k i} \\middle| \\mu\_0, \\kappa\_0\\right)\\right] p\\left(\\sigma\_i^{-2}\\middle| a\_0, b\_0\\right)
\\end{multline}
<img src="/assets/hierarchical_model_7.svg" style="margin: none; background: none; border: none; box-shadow: none;">



<section data-background="assets/stan_splash.png">
  <h2>Bayesian Inference in Stan</h2>
  <p>Building, tweaking, enhancing, and enriching models easily without having to think about the implementation</p>
</section>



<section>
  <h3>Stan</h3>
  <ol>
    <li><p>Imperative probabilistic programming language</p></li>
    <li><p>Automatic differentiation for HMC</p></li>
    <li><p>Adaptation routines</p></li>
    <li><p>R, Python, MATLAB, Julia, Stata, and command line interfaces</p></li>
  </ol>
</section>



### Program Blocks of A Stan Program

Declare data and parameter variables, define the log-posterior:

```
data { }
transformed data { }
parameters { }
transformed parameters { } 
model { }
generated quantities { }
```



### The `data` Block

Reading in information from an external source:

```
data {
    int<lower=1> K;              // number of classes
    int<lower=1> D;              // number of features
    int<lower=0> N;              // number of labelled observations
    int<lower=1,upper=K> c[N];   // classes for labelled observations
    vector[D] x[N];              // features for labelled observations
    vector<lower=0>[K] alpha;    // class concentration
    real mu0;                    // prior mean
    real<lower=0> kappa0;        // prior sample size
    real<lower=0> a0;            // shape
    real<lower=0> b0;            // rate
}
```



### The `transformed` `data` Block

Manipulate the external information once:

```
transformed data { }
```

We won't need it



### The `parameters` Block

Define the things we are going to sample from:

```
parameters {
    simplex[K] pi;                   // class prevalence
    vector[D] mu[K];                 // means of features
    vector<lower=0>[D] invsigmasqr;  // inverse variances of features
}
```



### The `transformed` `parameters` Block

Process parameters before computing the posterior:

```
transformed parameters {
    vector<lower=0>[D] sigma;    // scales of features
    for (i in 1:D)
        sigma[i] <- inv_sqrt(invsigmasqr[i]);
}
```

Parameters defined here are not sampled by the Markov chain



### The `model` Block

Define the posterior:

```
model {
    pi ~ dirichlet(alpha);       // class prevalence prior
    for (n in 1:N)
        c[n] ~ categorical(pi);  // class indicator prior
    for (k in 1:K)
        mu[k] ~ normal(mu0, sigma/sqrt(kappa0));
    for (i in 1:D)
        invsigmasqr[i] ~ gamma(a0, b0);
                                 // base measures
    for (n in 1:N)
        x[n] ~ normal(mu[c[n]], sigma);
                                 // generative model
}
```


### The `generated` `quantities` Block

Produce random samples, e.g. to validate the model with pseudo-data:

```
generated quantities {
    vector[K] z[Np];
    vector[K] sm[Np];
    int<lower=1,upper=K> cp[Np];
    for (np in 1:Np) {
        for (k in 1:K)
            z[np, k] <- normal_log(xp[np], mu[k], sigma) \
                + categorical_log(k, pi);
        sm[np] <- softmax(z[np]);
        cp[np] <- categorical_rng(sm[np]);
    }
}
```



## Demo



### Further Content Ingestion

[Stan User Guide and Reference Manual v2.9.0](https://github.com/stan-dev/stan/releases/download/v2.9.0/stan-reference-2.9.0.pdf)
  
  - Tons of example programs
  - Introduction to Bayesian statistics
  - Language reference
  - Discussion of HMC and NUTS algorithms

[Example program repository](stan example programs)

[Michael Betancourt's YouTube videos](https://www.youtube.com/watch?v=pHsuIaPbNbY)

[Andrew Gelman's book](bayesian data analysis gelman bibtex)



### Logistic Regression in Church

```
(define xs '(-10 -5 2 6 10))
(define labels '(#f #f #t #t #t))

(define samples
  (mh-query 
   1000 10
   (define m (gaussian 0 1))
   (define b (gaussian 0 1))
   (define sigma-squared (gamma 1 1))
   (define (y x)
     (gaussian (+ (* m x) b) sigma-squared))
   (define (sigmoid x)
     (/ 1 (+ 1 (exp (* -1 (y x))))))
   (sigmoid 8)
   (all
    (map (lambda (x label) (equal? (flip (sigmoid x) label) label))
         xs
         labels))))

(density samples "P(label=#t) for x=8" #t)
```

See also http://forestdb.org/models/logistic-regression.html

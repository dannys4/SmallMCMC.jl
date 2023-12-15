function GMMTest(rng)
    n_comps, dim = 3, 3
    means = [randn(rng, dim) for _ in 1:n_comps]
    sqrt_covs = [randn(rng, dim, dim) for _ in 1:n_comps]
    covs = [c*c' for c in sqrt_covs]
    wts = rand(rng, n_comps)
    wts /= sum(wts)
    gmm = MixtureModel([MvNormal(m,c) for (m,c) in zip(means, covs)], wts)
    C_0 = Float64.(I(dim))
    mcmc = MCMC_Adaptive_RWMH((x,p)->logpdf(gmm,x), C_0)
    N_chain, burnin = 50_000, 2_000
    chain = mcmc_sample(mcmc, zeros(dim), N_chain, verbose = false)
    chain = chain[:,burnin:end]
    true_mean, est_mean = mean(gmm), mean(chain,dims=2)
    true_cov, est_cov = cov(gmm), cov(chain,dims=2)
    mean_err = norm(true_mean - est_mean)/norm(true_mean)
    cov_err = norm(true_cov - est_cov)/norm(true_cov)
    tol = 20/sqrt(N_chain - burnin)
    @test mean_err < tol
    @test cov_err < dim*tol
end
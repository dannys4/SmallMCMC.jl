"Simple autocorrelation function"
function autocorrelation(chain::Matrix, max_corr_length=size(chain,2)÷10, y_bar = mean(chain, dims=2); verbose=true)
    ac = zeros(size(chain, 1), max_corr_length)
    chain_c = chain .- y_bar
    chain_v = sum(chain_c.^2, dims=2)
    prog = Progress(max_corr_length, 1, "Computing Chain Autocorrelation...", enabled=verbose)
    for i in 1:max_corr_length
        ac[:,i] = sum(chain_c[:,1:end-i] .* chain_c[:,i+1:end], dims=2) ./ chain_v
        next!(prog)
    end
    ac
end
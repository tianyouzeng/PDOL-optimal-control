using GaussianRandomFields
using MAT
using Plots

N = 2048
m = 64
padding = 64
f = zeros(N, m, m, m)

cov = CovarianceFunction(3, Gaussian(5, σ = 0.3))
pts = range(0, stop=1, length=m)
grf = GaussianRandomField(cov, CirculantEmbedding(), pts, pts, pts, minpadding=padding)

for i = 1 : N
    f[i, :, :, :] = sample(grf)
    if i % 64 == 0
        println(i)
    end
end

matwrite("bp_gradadj_train_f.mat", Dict("f" => f); compress = true)

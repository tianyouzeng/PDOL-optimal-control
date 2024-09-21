using GaussianRandomFields
using MAT
using Plots

N = 2048
m = 64
padding = 64
u = zeros(N, m, m, m)

cov = CovarianceFunction(3, Gaussian(0.5, σ = 150.0))
pts = range(0, stop=1, length=m)
Z = [x * (x-1) * y * (y-1) for x in pts, y in pts, t in pts]
grf = GaussianRandomField(cov, CirculantEmbedding(), pts, pts, pts, minpadding=padding)

thre_float = 5.0
thre = thre_float * ones(m, m, m)
fac = 3.0
u_a = -1.0 * ones(m, m, m)
u_b = 2.0 * ones(m, m, m)

for i = 1 : N
    uf = Z .* sample(grf)
    uf[(-thre.<uf) .& (uf.<thre)] = zeros(size(uf[(-thre.<uf) .& (uf.<thre)]))
    uf[uf.>thre] = (uf[uf.>thre] .- thre_float) ./ (fac + 1)
    uf[uf.<-thre] = (uf[uf.<-thre] .+ thre_float) ./ (fac + 1)
    uf[uf .> u_b] = u_b[uf .> u_b]
    uf[uf .< u_a] = u_a[uf .< u_a]
    uf = 10.0 .* uf

    u[i, :, :, :] = uf
    if i % 64 == 0
        println(i)
    end
end

matwrite("sp_gradadj_train_u.mat", Dict("u" => u); compress = true)

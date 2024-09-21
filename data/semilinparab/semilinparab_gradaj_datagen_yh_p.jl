using GaussianRandomFields
using MAT
using Plots

# Part 1: Generate the bilinear coefficient

N = 2048
m = 64
padding = 70
u = zeros(N, m, m, m)

cov = CovarianceFunction(3, Gaussian(0.5, σ = 5.0))
pts = range(0, stop=1, length=m)
grf = GaussianRandomField(cov, CirculantEmbedding(), pts, pts, pts, minpadding=padding)

Z = [x * (x-1) * y * (y-1) * t^(1/2.5) for x in pts, y in pts, t in pts]

for i = 1 : N
    uf = Z .* sample(grf)

    u[i, :, :, :] = uf
    if i % 64 == 0
        println(i)
    end
end

matwrite("sp_gradadj_train_yh.mat", Dict("yh" => u); compress = true)


# Part 2: Generate the rhs coefficient

N = 2048
m = 64
padding = 70
u = zeros(N, m, m, m)

cov = CovarianceFunction(3, Gaussian(0.5, σ = 5.0))
pts = range(0, stop=1, length=m)
grf = GaussianRandomField(cov, CirculantEmbedding(), pts, pts, pts, minpadding=padding)

Z = [x * (x-1) * y * (y-1) * (1-t)^(1/2.5) for x in pts, y in pts, t in pts]
yd = [exp(-20.0 * ((x - 0.2)^2 + (y - 0.2)^2 + (t - 0.2)^2)) + exp(-20.0 * ((x - 0.7)^2 + (y - 0.7)^2 + (t - 0.9)^2)) for x in pts, y in pts, t in pts]

for i = 1 : N
    uf = Z .* sample(grf) - yd

    u[i, :, :, :] = uf
    if i % 64 == 0
        println(i)
    end
end

matwrite("sp_gradadj_train_f.mat", Dict("f" => u); compress = true)

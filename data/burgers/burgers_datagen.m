%% Generate training data for stationary Burger equation

N = 512;
len_scale = 0.2;
output_scale = 0.2;
m = 101; % Resolution of output functions
num = 5000;
global vq_global zq_global y_global dydx_global; % passing vq to bcfcn causes a vector dimension bug

u = zeros(num, m);
y = zeros(num, m);
dydx = zeros(num, m);
d2ydx2 = zeros(num, m);
z = zeros(num, m);
x = linspace(0, 1, N);
p = zeros(num, m);
dpdx = zeros(num, m);
d2pdx2 = zeros(num, m);

ker = RBF_cond(x, len_scale, output_scale);
jitter = 1e-10;
L = chol(ker + jitter*eye(N-2));

ker_adj = RBF(x, len_scale, output_scale);
L_adj = chol(ker_adj + jitter*eye(N));

for i = 1 : num

    gp_sample = zeros(N, 1);
    gp_sample(2:end-1) = L' * normrnd(0, 1, N-2, 1);
    
    xq = linspace(0, 1, m);
    vq = interp1(x, gp_sample, xq, 'spline');
    vq([1, end]) = [0, 0]; % avoid rounding error
    vq = min(vq, 0.3 * ones(size(vq)));
    vq_global = vq;

    gp_sample_adj = L_adj' * normrnd(0, 1, N, 1);
    
    xq = linspace(0, 1, m);
    zq = interp1(x, gp_sample_adj, xq, 'spline');
    % zq([1, end]) = [0, 0]; % avoid rounding error
    zq_global = zq;
    
    xmesh = linspace(0, 1, 101);
    solinit = bvpinit(xmesh, @guess);
    options = bvpset('RelTol', 1e-8);
    sol = bvp5c(@bvpfcn, @bcfcn, solinit, options);

    u(i, :) = vq;
    y(i, :) = interp1(sol.x, sol.y(1, :), xq, 'spline');
    y(i, [1, end]) = [0, 0]; % avoid rounding error
    dydx(i, :) = interp1(sol.x, sol.y(2, :), xq, 'spline');
    d2ydx2(i, :) = interp1(sol.x, sol.idata.yp(2, :), xq, 'spline');
    z(i, :) = zq;
    y_global = y(i, :);
    dydx_global = dydx(i, :);

    sol_adj = bvp5c(@bvpfcn_adj, @bcfcn_adj, solinit, options);

    p(i, :) = interp1(sol_adj.x, sol_adj.y(1, :), xq, 'spline');
    p(i, [1, end]) = [0, 0];
    dpdx(i, :) = interp1(sol_adj.x, sol_adj.y(2, :), xq, 'spline');
    d2pdx2(i, :) = interp1(sol_adj.x, sol_adj.idata.yp(2, :), xq, 'spline');

    if mod(i, 100) == 0
        disp(i);
    end
end

x = xq;
save("burgers_data_train.mat", "x", "u", "y", "dydx", "d2ydx2", "z", "p", "dpdx", "d2pdx2");

%% Covariance matrices

% Unconditioned covariance matrix
function ker = RBF(x, lenscale, outscale)
    diff = abs((x - x') ./ lenscale);
    ker = outscale * exp(-0.5 * diff.^2);
end

% Covariance matrix conditioned on zero Direchlet BC
function ker = RBF_cond(x, lenscale, outscale) % x >= 3 and 0, 1 \in x
    diff = abs((x - x') ./ lenscale);
    ker_origin = outscale * exp(-0.5 * diff.^2);
    ker = ker_origin(2:end-1, 2:end-1) - ker_origin(2:end-1, [1, end]) * inv(ker_origin([1, end], [1, end])) * ker_origin([1, end], 2:end-1);
end

%% Functions for PDE solver

function yx = bvpfcn(x, y)
    nu = 1 / 12;
    global vq_global;
    value = interp1(linspace(0, 1, 101), vq_global, x, 'spline');
    yx = [y(2); (y(1) .* y(2) - value) ./ nu];
end

function res = bcfcn(ya, yb)
    res = [ya(1); yb(1)];
end

function g = guess(x)
    g = [0;0];
end

function px = bvpfcn_adj(x, p)
    global zq_global y_global;
    nu = 1/12;
    value_y = interp1(linspace(0, 1, 101), y_global, x, 'spline');
    % value_dydx = interp1(linspace(0, 1, 101), dydx_global, x, 'spline');
    value_z = interp1(linspace(0, 1, 101), zq_global, x, 'spline');
    px = [p(2); (-value_y * p(2) - value_z) / nu];
end

function res = bcfcn_adj(pa, pb)
    res = [pa(1); pb(1)];
end
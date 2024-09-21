n = 2048;
start = 1;
m = 64;

gridpoints = linspace(0, 1, m);
[xmesh, ymesh] = meshgrid(gridpoints, gridpoints);
xq = gridpoints;
yq = gridpoints;
tq = linspace(0, 1, m);
tlist = tq;
[X, Y] = meshgrid(xq, yq);
iT = 1:length(tlist);

[xm, ym, tm] = meshgrid(gridpoints, gridpoints, gridpoints);
load("sp_gradadj_train_yh.mat");
load("sp_gradadj_train_f.mat");

s = zeros(n - start + 1, m, m, m);

R1 = [3,4,0,1,1,0,1,1,0,0]';
g = decsg(R1);

for i = start:n
    disp(i);
    tic
    model = createpde();
    geometryFromEdges(model, g);
    applyBoundaryCondition(model,"dirichlet", "Edge", 1:model.Geometry.NumEdges, "u", 0);
    setInitialConditions(model, 0);
    yhi = squeeze(yh(i,:,:,:));
    Ryhi = yhi .* (yhi - 0.25) + (yhi - 0.25) .* (yhi + 1.0) + (yhi + 1.0) .* yhi;
    acoefffunc_short = @(location, state) acoefffunc(location, state, yhi);
    fcoefffunc_short = @(location, state) fcoefffunc(location, state, squeeze(f(i,:,:,:)));
    specifyCoefficients(model, "m", 0, "d", 1, "c", 1, "a", acoefffunc_short, "f", fcoefffunc_short);
    generateMesh(model, "Hmax", 0.03);
    tlist = tq;
    sresults = solvepde(model,tlist);
    
    [X, Y] = meshgrid(xq, yq);
    iT = length(tlist):-1:1;
    sinterp = interpolateSolution(sresults, X, Y, iT);
    sinterp = reshape(sinterp, [size(X), length(iT)]);

    s(i - start + 1,:,:,:) = sinterp;
    toc
end

save("sp_gradadj_train_y_play.mat", "s", '-v7.3');

function acoeff = acoefffunc(location, state, u)
    [x, y, t] = meshgrid(linspace(0, 1, 64), linspace(0, 1, 64), linspace(0, 1, 64));
    xq = location.x;
    yq = location.y;
    if isnan(state.time)
        acoeff = nan(size(xq));
    else
        tq = (1.0 - state.time) * ones(size(xq));
        acoeff = interp3(x, y, t, u, xq, yq, tq);
    end
end

function fcoeff = fcoefffunc(location, state, f)
    [x, y, t] = meshgrid(linspace(0, 1, 64), linspace(0, 1, 64), linspace(0, 1, 64));
    xq = location.x;
    yq = location.y;
    if isnan(state.time)
        fcoeff = nan(size(xq));
    else
        tq = (1.0 - state.time) * ones(size(xq));
        fcoeff = interp3(x, y, t, f, xq, yq, tq);
    end
end
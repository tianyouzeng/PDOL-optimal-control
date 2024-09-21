n = 5;
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
load("bp_cts_gradadj_train_u.mat");
load("bp_cts_gradadj_train_f.mat");

s = zeros(n - start + 1, m, m, m);

R1 = [3,4,0,1,1,0,1,1,0,0]';
g = decsg(R1);

for i = start : n
    disp(i);
    model = createpde();
    geometryFromEdges(model, g);
    applyBoundaryCondition(model,"dirichlet", "Edge", 1:model.Geometry.NumEdges, "u", 0);
    setInitialConditions(model, 0);
    mesh = generateMesh(model);
    acoefffunc_short = @(location, state) acoefffunc(location, state, squeeze(u(i,:,:,:)), xm, ym, tm);
    fcoefffunc_short = @(location, state) fcoefffunc(location, state, squeeze(f(i,:,:,:)), xm, ym, tm);
    specifyCoefficients(model, "m", 0, "d", 1, "c", 1, "a", acoefffunc_short, "f", fcoefffunc_short);

    tic
    sresults = solvepde(model,tlist);
    toc

    sinterp = interpolateSolution(sresults, X, Y, iT);
    sinterp = reshape(sinterp, [size(X), length(iT)]);

    s(i - start + 1,:,:,:) = sinterp;
end

save("bp_cts_train_y.mat", "s", '-v7.3');

function acoeff = acoefffunc(location, state, u, xlist, ylist, tlist)
    xq = location.x;
    yq = location.y;
    tq = state.time * ones(size(xq));
    % MATLAB PDE toolbox use the return value at t=nan to determine whether
    % using nonlinear solver or not
    if isnan(tq)  
        acoeff = nan(size(xq));
    else
        acoeff = interp3(xlist, ylist, tlist, u, xq, yq, tq);
    end
end

function fcoeff = fcoefffunc(location, state, f, xlist, ylist, tlist)
    xq = location.x;
    yq = location.y;
    tq = state.time * ones(size(xq));
    if isnan(tq)  
        fcoeff = nan(size(xq));
    else
        fcoeff = interp3(xlist, ylist, tlist, f, xq, yq, tq);
    end
end

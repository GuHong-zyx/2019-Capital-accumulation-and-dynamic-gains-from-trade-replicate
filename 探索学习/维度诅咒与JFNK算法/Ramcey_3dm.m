clear; clc; close all;

%% 参数
p = struct();
p.beta = 0.95;
p.alpha = [0.35; 0.25; 0.30];
p.delta = 0.08;
p.tol = 1e-6;
p.maxiter = 100;
p.T = 150;
p.z_old = 0.9;
p.z_new = 1.1;
p.min_c = 1e-10;
p.k_count = length(p.alpha);
p.z_path = p.z_new * ones(1, p.T);

%% 稳态
oldss = computess(p.z_old, p);
newss = computess(p.z_new, p);

fprintf('old steady state:\n');
disp(oldss);

fprintf('new steady state:\n');
disp(newss);

%% 初始猜测：资本线性过渡路径
k_path0 = zeros(p.k_count, p.T + 1);

for t = 1:p.T + 1
    w = (t - 1) / p.T;
    k_path0(:, t) = (1 - w) * oldss.k + w * newss.k;
end

% 未知量是 k(:,2:T)，固定 k(:,1) 和 k(:,T+1)
x = reshape(k_path0(:, 2:p.T), [], 1);

%% JFNK 主循环
Ffun = @(x) residual_x(x, oldss.k, newss.k, p);

for iter = 1:p.maxiter
    F = Ffun(x);
    Fnorm = norm(F);

    fprintf('iter = %3d, residual norm = %.4e\n', iter, Fnorm);

    if Fnorm < p.tol
        break;
    end

    % Jacobian-vector product by finite difference
    Jv = @(v) jacvec(Ffun, x, F, v);

    % Newton-Krylov: J dx = -F
    gmres_tol = min(1e-2, sqrt(Fnorm));
    max_gmres_iter = min(length(x), 80);

    [dx, flag] = gmres(Jv, -F, [], gmres_tol, max_gmres_iter);

    if flag ~= 0
        warning('GMRES did not fully converge, flag = %d', flag);
    end

    % 线搜索，保证消费为正且残差下降
    step = 1.0;
    success = false;

    for ls = 1:25
        x_try = x + step * dx;
        [k_try, c_try, y_try] = paths_from_x(x_try, oldss.k, newss.k, p);

        if min(c_try) > p.min_c
            F_try = Ffun(x_try);

            if norm(F_try) < (1 - 1e-4 * step) * Fnorm
                x = x_try;
                success = true;
                break;
            end
        end

        step = step / 2;
    end

    if ~success
        warning('Line search failed. Stop.');
        break;
    end
end

%% 得到最终路径
[k_path, c_path, y_path] = paths_from_x(x, oldss.k, newss.k, p);
F_final = Ffun(x);

fprintf('\nFinal residual norm = %.4e\n', norm(F_final));
fprintf('Minimum consumption = %.4e\n', min(c_path));

%% 简单可视化
time_k = 0:p.T;
time_c = 1:p.T;

figure;

subplot(2,2,1);
plot(time_k, k_path', 'LineWidth', 1.5);
xlabel('t');
ylabel('k_i');
title('Capital paths');
legend('k_1','k_2','k_3', 'Location', 'best');
grid on;

subplot(2,2,2);
plot(time_c, c_path, 'LineWidth', 1.5);
xlabel('t');
ylabel('c');
title('Consumption path');
grid on;

subplot(2,2,3);
plot(time_c, y_path, 'LineWidth', 1.5);
xlabel('t');
ylabel('y');
title('Output path');
grid on;

subplot(2,2,4);
plot(time_c(1:end-1), vecnorm(reshape(F_final, p.k_count, p.T - 1)), 'LineWidth', 1.5);
xlabel('t');
ylabel('Euler residual norm');
title('Euler residual by time');
grid on;

%% ===================== 函数区 =====================

function y = computey(k, z, p)
    y = z * prod(k .^ p.alpha);
end

function mp = computemp(k, z, p)
    y = computey(k, z, p);
    mp = p.alpha .* y ./ k;
end

function ss = computess(z, p)
    r = 1 / p.beta - 1 + p.delta;
    alpha_sum = sum(p.alpha);

    y = (z * prod((p.alpha ./ r) .^ p.alpha)) ^ (1 / (1 - alpha_sum));
    k = p.alpha .* y ./ r;
    c = y - p.delta * sum(k);

    ss = struct();
    ss.c = c;
    ss.y = y;
    ss.k = k;
end

function [k_path, c_path, y_path] = paths_from_x(x, k0, kT1, p)
    k_mid = reshape(x, p.k_count, p.T - 1);

    k_path = zeros(p.k_count, p.T + 1);
    k_path(:, 1) = k0;
    k_path(:, 2:p.T) = k_mid;
    k_path(:, p.T + 1) = kT1;

    c_path = zeros(1, p.T);
    y_path = zeros(1, p.T);

    for t = 1:p.T
        k_today = k_path(:, t);
        k_next = k_path(:, t + 1);

        y_path(t) = computey(k_today, p.z_path(t), p);
        c_path(t) = y_path(t) + (1 - p.delta) * sum(k_today) - sum(k_next);
    end
end

function F = residual_x(x, k0, kT1, p)
    [k_path, c_path, ~] = paths_from_x(x, k0, kT1, p);

    F_mat = zeros(p.k_count, p.T - 1);

    if min(c_path) <= p.min_c || min(k_path(:)) <= 0
        F = 1e8 * ones(p.k_count * (p.T - 1), 1);
        return;
    end

    for t = 1:p.T - 1
        k_next = k_path(:, t + 1);
        mp_next = computemp(k_next, p.z_path(t + 1), p);

        for i = 1:p.k_count
            lhs = c_path(t + 1) / c_path(t);
            rhs = p.beta * (mp_next(i) + 1 - p.delta);
            F_mat(i, t) = log(lhs) - log(rhs);
        end
    end

    F = reshape(F_mat, [], 1);
end

function Jv = jacvec(Ffun, x, F, v)
    h = 1e-6 * (1 + norm(x)) / max(norm(v), 1);
    Jv = (Ffun(x + h * v) - F) / h;
end
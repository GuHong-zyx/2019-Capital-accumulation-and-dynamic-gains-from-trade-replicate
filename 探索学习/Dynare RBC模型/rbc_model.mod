var c k y i a;  //内生变量
varexo e; //外生冲击

parameters beta alpha delta rho sigma; //参数

beta  = 0.99;
alpha = 0.36;
delta = 0.025;
rho   = 0.95;
sigma = 0.01;

model; //运动方程
  y = exp(a)*k(-1)^alpha;
  y = c + i;
  k = i + (1-delta)*k(-1);
  1/c = beta*(1/c(+1))*(alpha*exp(a(+1))*k^(alpha-1) + 1 - delta);
  a = rho*a(-1) + e;
end;

initval; // 求出初始稳态
  a = 0;
  k = ((1/beta - 1 + delta)/alpha)^(1/(alpha-1));
  y = k^alpha;
  i = delta*k;
  c = y - i;
end;

steady;
check;

shocks;
  var e = sigma^2;
end;

stoch_simul(order=1, irf=40);

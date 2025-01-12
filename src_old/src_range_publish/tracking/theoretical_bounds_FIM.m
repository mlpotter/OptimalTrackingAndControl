close all; clear all; clc
%% define the radar parameters
c = 299792458
fc = 1e8;
Gt = 200;
Gr = 200;
lam = c / fc
rcs = 1;
L = 1;

%calculate Pt such that I achieve SNR=x at distance R=y
R_desired = 500

Pt = 1000
K = Pt * Gt * Gr * lam ^ 2 * rcs / L / (4 * pi) ^ 3
Pr = K / (R_desired ^ 4)

% get the power of the noise of the signalf
SNR=0

% white noise variance on the measurement
sigmaW = sqrt(Pr/ (10^(SNR/10)))

% constant from radar parameters and noise on the match filter
C = c^2 * sigmaW^2 / (pi^2 * 8 * fc^2) * 1/K
%% define the process noise parameters nad measurement noise parameters
dt= 0.05;

sigma_q = 3;

sigma_0 = 5;

A = [1 0 0 dt 0 0;
     0 1 0 0 dt 0;
     0 0 1 0 0 dt;
     0 0 0 1 0 0;
     0 0 0 0 1 0;
     0 0 0 0 0 1];

Q =  [
    (dt ^ 4) / 4, 0, 0, (dt ^ 3) / 2, 0, 0;
    0, (dt ^ 4) / 4, 0, 0, (dt ^ 3) / 2, 0;
    0, 0, (dt ^ 4) / 4, 0, 0, (dt ^ 3) / 2;
    (dt ^ 3) / 2, 0, 0, (dt ^ 2), 0, 0;
    0, (dt ^ 3) / 2, 0, 0, (dt ^ 2), 0;
    0, 0, (dt ^ 3) / 2, 0, 0, (dt ^ 2)] * sigma_q ^ 2;


radar_pos = [0 0 25;
            80 80 25;
            -15 30 5;
            0 30 5];

N_radar = size(radar_pos,1);
N_dim = size(A,1);


x0 = [[5 5 5 0.1 0.3 0]].';
P0 = eye(N_dim)*sigma_0^2;

N = 2000
nMC = 250

% y = zeros(N_radar, N);

%% MC trials
% extra memory for MC
x_record = zeros(N_dim,N+1,nMC);
x_ckf_record = zeros(N_dim,N+1,nMC);
x_error1_MC = zeros(N+1, nMC);

x = zeros(N_dim,N+1);

for iMC = 1:nMC
    iMC

    x = zeros(N_dim,N+1);

    process_noise = mvnrnd(zeros(N_dim,1),Q,N).';
    % generate noisy measurements based on the target true state + noise
    for n = 1:N
        x(:,n+1) = TransitionFn(x(:,n),A) + process_noise(:,n);
    end
    x(:,1) = x0;

    y = zeros(N_radar, N);
    % y = zeros(3, N);
    
    x_ckf = zeros(N_dim,N+1);
    x_error = zeros(N_dim,N+1);
   

    x_ckf(:,1) = x0 + randn(N_dim,1)*sigma_0;

    ckf = trackingCKF(StateTransitionFcn=@TransitionFn, ...
        MeasurementFcn= @MeasurementFn, ...
        HasAdditiveProcessNoise=true, ...
        ...
        HasAdditiveMeasurementNoise=true, ...
        state=x_ckf(:,1));

    ckf.ProcessNoise = Q;
    
    x_error(:,1) = (x0-x_ckf(:,1));

    % generate noisy measurements based on the target true state + noise
    for n = 1:N
        R_n = C*diag(RangesFn(x(:,n+1),radar_pos).^4);

        y(:,n) = MeasurementFn(x(:,n+1) ,radar_pos) + mvnrnd(zeros(N_radar,1),R_n).'; 
        % y(:,n) = x(1:3,n+1) + measurement_noise(:,n);
    end
    
    % get the CKF errors and tracking
    for n = 1:N
        [xPred,pPred] = predict(ckf,A);

        R_n = C*diag(RangesFn(xPred,radar_pos).^4);

        ckf.MeasurementNoise = R_n ;

        [xCorr,pCorr] = correct(ckf,y(:,n),radar_pos);
    
        x_ckf(:,n+1) = xCorr;
    
        x_error(:,n+1) = (xCorr-x(:,n+1));
    end

    x_error1_MC(:,iMC) = x_error(1,:);

    % record
    if iMC == nMC
        x_record(:,:,iMC) = x;
        x_ckf_record(:,:,iMC) = x_ckf;
    end

end

%%
J = zeros(N_dim,N_dim,N+1);
J(:,:,1) = P0^(-1);
bound = zeros(N_dim,N_dim,N+1);
bound(:,:,1) = P0;
for n = 1: N
    standard_fim = STANDARD_FIM(x_record(:,n+1,nMC),radar_pos,C);
    % H = [eye(3), zeros(3,3)];
    J(:,:,n+1) = (Q + A*J(:,:,n)^(-1)*A')^(-1) + ...
            standard_fim;
    bound(:,:,n+1) = J(:,:,n+1)^(-1);
end
    
%%
figure
x_plot = x_record(:,:,nMC);
plot(x_plot(1,:),x_plot(2,:),'bo-')
hold on
x_ckf_plot = x_ckf_record(:,:,nMC);
plot(x_ckf_plot(1,:),x_ckf_plot(2,:),'g.-')
plot(radar_pos(:,1),radar_pos(:,2),'ro','MarkerSize',15,'MarkerFaceColor','r')
legend(["True Trajectory","CKF","Radars"])
%% RMSE
RMSE = zeros(1, N);
for n = 1:N+1
    RMSE(n) = sqrt(x_error1_MC(n,:)*x_error1_MC(n,:)'/nMC);
end
figure;
plot(1:N+1, RMSE)
hold on
bound1 = bound(1,1,:);
plot(1:N+1, sqrt(bound1(:)))
legend('RMSE','PCRB')
%%

function [state_next] = TransitionFn(state_cur,A)

    state_next =  A*state_cur;

end

function [standard_fim] = STANDARD_FIM(state_cur,radar_pos,C)
    target_pos = state_cur(1:3);
    
    diff = [target_pos-radar_pos.' ; zeros(3,size(radar_pos,1))];
    ranges = vecnorm(target_pos-radar_pos.',2,1);

    coef = sqrt(4./(C*ranges.^6) + 8./(ranges.^4));

    standard_fim =  (diff.*coef) * (diff.*coef)';
end

function ranges = RangesFn(state_cur,radar_pos)
    target_pos = state_cur(1:3);
    
    ranges = vecnorm(target_pos-radar_pos.',2,1);
end

function [y] = MeasurementFn(state_cur,radar_pos)
    target_pos = state_cur(1:3);

    ranges = vecnorm(target_pos-radar_pos.',2,1);

    y = 2*ranges.';

    % y = target_pos;

end

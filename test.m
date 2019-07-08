% Whiteboard Estimator
% sleutene 07/2019
clear all;
close all;

%% Parameters
lim_x = [-1,1]; % limits of whiteboard [m]
lim_z = [-1,1]; % limits of whiteboard [m]
N=100; % number of iterations
sigma_r = 0.01; % measurement stdev
sigma_q_r = 0.001; % process noise [m/sqrt(s)]
sigma_q_phi = 0.001; % process noise [rad/sqrt(s)]
sigma_q_psi = 0.001; % process noise [rad/sqrt(s)]
dt = 0.05; % 20 Hz

%% Randomised ground truth state
r_W_true = [1;1;1];
psi_true = pi/8;
phi_true = pi/16;
C_WT_true = computeC(phi_true, psi_true);
x_true = [r_W_true; phi_true; psi_true];

%% Initial state and covariance
P = diag([0.1, 0.1, 0.1, 0.1, 0.1]);
x = x_true + [0.1;0.1;0.1;0.1;0.1];
C_WT = computeC(x(4), x(5));

%% Initial plot
figure(1);
hold on
axis equal
grid on
plotAxes(eye(3), [0;0;0], 0.1);
plotAxes(C_WT_true, x_true(1:3), 0.4);
[h(1), h(2), h(3)] = plotAxes(C_WT, x(1:3), 0.6);

%% Run the filter
R=diag([sigma_r^2, sigma_r^2, sigma_r^2]);
for i=1:N
    % generate randomised measurement
    r_tilde_W = C_WT_true*[random('unif',...
        lim_x(1),lim_x(2));0;random('unif',lim_z(1),lim_z(2))]...
        + r_W_true + chol(R)*randn(3,1);
    % plot the measurement
    plot3(r_tilde_W(1),r_tilde_W(2),r_tilde_W(3),'*b')
    
    % prediction
    % nothing todo for x (assumed static)
    P = P + dt*diag([sigma_q_r^2, sigma_q_r^2, sigma_q_r^2, ...
        sigma_q_phi^2, sigma_q_psi^2]); % process noise
    
    % update
    C_WT = computeC(x(4), x(5));
    delta_r = (r_tilde_W-x(1:3));
    phi = x(4);
    psi = x(5);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % use this to compute H entries:
    % syms phi psi r1 r2 r3 real;
    % C_z = [ cos(psi), sin(psi), 0;...
    %        -sin(psi), cos(psi), 0;...
    %         0, 0, 1];
    % C_x = [1, 0, 0
    %        0,  cos(phi), sin(phi);...
    %        0, -sin(phi), cos(phi);];
    % C_WT = C_z * C_x;
    % r=[r1;r2;r3];
    % simplify(jacobian([0,1,0]*C_WT'*r,[phi;psi]))
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    H_phi = delta_r(3)*cos(phi) - delta_r(2)*cos(psi)*sin(phi) ...
        + delta_r(1)*sin(phi)*sin(psi);
    H_psi = -cos(phi)*(delta_r(1)*cos(psi) - delta_r(2)*sin(psi));
    y = [0, 1, 0]*C_WT'*delta_r; % residual
    H = [[0, 1, 0]*C_WT', H_phi, H_psi]; % measurement Jacobian
    S = H*P*H' + [0, 1, 0]*C_WT'*R*C_WT*[0, 1, 0]'; % residual covariance
    K = P*H'*inv(S); % Kalman gain
    delta_x = K*y; % state change
    % Not strictly needed, but makes the result smooth in the plane ?
    % constrain posiion update into normal direction:
    s = (C_WT*[0;1;0])'*delta_x(1:3); % projection along normal direction
    delta_x(1:3) = s*C_WT*[0;1;0]; % force normal direction pos. change
    x = x + delta_x; % actual state update
    P = P - K*H*P; % state covariance update
    
    % plotting
    delete(h(1));
    delete(h(2));
    delete(h(3));
    [h(1), h(2), h(3)] = plotAxes(C_WT, x(1:3), 0.6);
    drawnow
end

%% Helper functions
% compute orientation
function C_WT = computeC(phi, psi)
    C_z = [ cos(psi), sin(psi), 0;...
                -sin(psi), cos(psi), 0;...
                          0, 0, 1];
    C_x = [1, 0, 0
           0,  cos(phi), sin(phi);...
           0, -sin(phi), cos(phi);];
    C_WT = C_z * C_x;
end
% plot coloured axes
function [h1,h2,h3] = plotAxes(C_WT, r_WT, scale)
    p2 = r_WT+scale*C_WT(:,1);
    h1=plot3([r_WT(1);p2(1)], [r_WT(2);p2(2)], [r_WT(3);p2(3)],'-r');
    p2 = r_WT+scale*C_WT(:,2);
    h2=plot3([r_WT(1);p2(1)], [r_WT(2);p2(2)], [r_WT(3);p2(3)],'-b');
    p2 = r_WT+scale*C_WT(:,3);
    h3=plot3([r_WT(1);p2(1)], [r_WT(2);p2(2)], [r_WT(3);p2(3)],'-g');
end



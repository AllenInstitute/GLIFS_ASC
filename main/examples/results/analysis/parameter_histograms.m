%% Produces Figure S4
% figure stored to {task}_{n}_paramhistogram.svg
clear all
fig = figure('Position', get(0, 'Screensize'));
fig.Renderer='Painters';

% TODO: Change below two lines
task = "sine"; % sine or lmnist
n = "glifr_lheta";

fontsize = 24;

fincolor = "#40B0A6";
initcolor = "#E1BE6A";

nbins = 20;
% MEM: thresh, k_m
% ASC: k, r, amp
param_names = ["thresh (mV)", "k*_m (1/ms)", "a_j (pA)", "r*_j", "k*_j (1/ms)"];
params = ["thresh", "trans_k_m", "a_j", "trans_r_j", "trans_k_j"];
for param = params
    param_trajectory_init = readmatrix("./../glifr_homa_for_init/lmnist_glifr_homa_" + param + "_param-trajectory.csv");
%     d = dir("./../" + task + "/" + "glifr_lheta" + "/*/*/*/" + param + "_param-trajectory.csv");
%     filename = d(1).folder + "\" + d(1).name;
%     param_trajectory = readmatrix(filename);
%     param_trajectory_init = param_trajectory(size(param_trajectory, 1), :);
    
    d = dir("./../" + task + "/" + n + "/*/*/*/" + param + "_param-trajectory.csv");
    filename = d(1).folder + "\" + d(1).name;
    param_trajectory = readmatrix(filename);
    param_trajectory_final = param_trajectory(size(param_trajectory, 1), :);
    
    idx = find(strcmp(params, param));
    h(idx) = subplot(3,2,idx);
    hh = histogram(param_trajectory_final, nbins, 'FaceAlpha', 1, 'FaceColor', fincolor);
    hold on
    histogram(param_trajectory_init, hh.BinEdges, 'FaceAlpha', 0.8, 'FaceColor', initcolor);
    Ylm = ylim;
    ylim([Ylm(1), 1.1 * Ylm(2)]);
    xlabel(param_names(idx), 'FontSize', fontsize);
    ylabel("count", 'FontSize', fontsize);
    set(gca,'FontSize', fontsize-8);
end
pos = get(h,'Position');
new = mean(cellfun(@(v)v(1),pos(1:2)));
set(h(5),'Position',[new,pos{end}(2:end)])

saveas(fig, task + "_" + n + "_paramhistogram.svg",'svg');
close(fig);
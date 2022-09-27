%% Produces Figure 9 and Figure S5
% Figures stored to {task}_chscores.svg, {task}_paramclusters.svg,
% {task}_ficurveclusters.svg

clear all

% TODO: Change beow two lines
task = "nmnist"; % smnist or pattern
n = "glifr_lheta";

dt = 1;
fontsize = 24;
linewidth = 2;


d = dir("./../" + task + "-post/" + n + "/*/*/*/chscores.csv")
filename = d(1).folder + "\" + d(1).name;
ch_scores = readmatrix(filename);

d = dir("./../" + task + "-post/" + n + "/*/*/*/cluster_labels.csv")
filename = d(1).folder + "\" + d(1).name;
clusters_labels = readmatrix(filename);

d = dir("./../" + task + "-post/" + n + "/*/*/*/learnedparams.csv")
filename = d(1).folder + "\" + d(1).name;
parameters = readmatrix(filename);

d = dir("./../" + task + "-post/" + n + "/*/*/*/isyns.csv")
filename = d(1).folder + "\" + d(1).name;
isyns = readmatrix(filename);

d = dir("./../" + task + "-post/" + n + "/*/*/*/frates.csv")
filename = d(1).folder + "\" + d(1).name;
frates = readmatrix(filename);
isyns_init = readmatrix("./../glifr_homa_for_init/nmnist_glifr_homa_isyns.csv");
frates_init = readmatrix("./../glifr_homa_for_init/nmnist_glifr_homa_frates.csv");
    
% PLOT CH SCORES
fig = figure('Position', get(0, 'Screensize'));
fig.Renderer='Painters';
scatter(ch_scores(:, 1), ch_scores(:, 2), 200, 'k', 'filled');
xlabel('# clusters', 'FontName', 'helvetica', 'FontSize', fontsize);
ylabel('CH index', 'FontName', 'helvetica', 'FontSize', fontsize);
set(gca,'FontSize', fontsize);
saveas(fig, task + "_chscores.svg",'svg');
close(fig);


num_neurons = size(parameters, 1);
% PLOT PARAMETER SCATTERPLOTS
fig = figure('Position', get(0, 'Screensize'));
fig.Renderer='Painters';

colors_rbg = [51, 34, 136;
    68, 170, 153;
    221, 204, 119;
    204, 102, 119;
    114,184,3] ./ 256;

subplot(2,2,1)
for i = 1:size(parameters, 1)
    scatter(parameters(i,1), parameters(i,2), 'filled', 'MarkerFaceColor', colors_rbg(clusters_labels(i) + 1,:), 'MarkerFaceAlpha',.2);
    hold on;
end
scatter([1], [0.05], 'k', 'filled', 'MarkerFaceAlpha', 1);
xlabel('thresh (mV)', 'FontName', 'helvetica', 'FontSize', fontsize);
ylabel('k_m (1/ms)', 'FontName', 'helvetica', 'FontSize', fontsize);
% xlim([0,1.5]);
% ylim([0,0.15]);
set(gca,'FontSize', fontsize);
colormap(colors_rbg)

subplot(2,2,2)
fig.Renderer='Painters';
for i = 1:size(parameters, 1)
    scatter(parameters(i,3), parameters(i,4), 'filled', 'MarkerFaceColor', colors_rbg(clusters_labels(i) + 1,:), 'MarkerFaceAlpha',.2);
    hold on;
end
scatter([0], [0], 'k', 'filled', 'MarkerFaceAlpha', 1);
xlabel('r_1', 'FontName', 'helvetica', 'FontSize', fontsize);
ylabel('r_2', 'FontName', 'helvetica', 'FontSize', fontsize);
set(gca,'FontSize', fontsize);

subplot(2,2,3)
fig.Renderer='Painters';
for i = 1:size(parameters, 1)
    scatter(parameters(i,5), parameters(i,6), 'filled', 'MarkerFaceColor', colors_rbg(clusters_labels(i) + 1,:), 'MarkerFaceAlpha',.2);
    hold on;
end
scatter([0.1/dt], [0.1/dt], 'k', 'filled', 'MarkerFaceAlpha', 1);
xlabel('k_1 (1/ms)', 'FontName', 'helvetica', 'FontSize', fontsize);
ylabel('k_2 (1/ms)', 'FontName', 'helvetica', 'FontSize', fontsize);
set(gca,'FontSize', fontsize);

subplot(2,2,4)
fig.Renderer='Painters';
for i = 1:size(parameters, 1)
    scatter(parameters(i,7), parameters(i,8), 'filled', 'MarkerFaceColor', colors_rbg(clusters_labels(i) + 1,:), 'MarkerFaceAlpha',.2);
    hold on;
end
scatter([0], [0], 'k', 'filled', 'MarkerFaceAlpha', 1);
% hold on
% h=fill([-0.01,0.01,0.01,-0.01],[-0.01,-0.01,0.01,0.01],'red');
% h.FaceAlpha=0.3;
xlabel('a_1 (pA)', 'FontName', 'helvetica', 'FontSize', fontsize);
ylabel('a_2 (pA)', 'FontName', 'helvetica', 'FontSize', fontsize);
set(gca,'FontSize', fontsize);
saveas(fig, task + "_paramclusters.svg",'svg');
close(fig);

fig = figure('Position', get(0, 'Screensize'));
fig.Renderer='Painters';
colors = ["#332288", "#44AA99", "#DDCC77", "#CC6677", "#72B803", "#AA4499", "#882255", "#72B803", "#109EC4", "#4DB8F6", "#4E1D87"];

% PLOT f-I CURVES
for i=1:size(frates, 2)
%     if clusters_labels(i) ~= 4
%         continue;
%     end
    hline = plot(isyns, frates(:,i), 'LineWidth', linewidth, 'color', colors(clusters_labels(i)+1))
    for i=1:length(hline)
        hline(i).Color = [hline(i).Color 0.5];  % alpha=0.1
    end
    hold on;
end
hold on
for i=1:size(frates_init, 2)
    plot(isyns_init, frates_init(:,i), 'k', 'LineWidth', linewidth)
    hold on
end

p1 = plot(nan, nan, 'LineWidth', linewidth, 'color', colors(1));
p2 = plot(nan, nan, 'LineWidth', linewidth, 'color', colors(2));
p3 = plot(nan, nan, 'LineWidth', linewidth, 'color', colors(3));
p4 = plot(nan, nan, 'LineWidth', linewidth, 'color', colors(4));
p5 = plot(nan, nan, 'LineWidth', linewidth, 'color', colors(5));
ps = [p1, p2, p3, p4, p5];
ls = {"A", "B", "C", "D", "E"};
legend(ps, ls, 'Location', 'eastoutside');

set(gca,'FontSize', fontsize);
xlabel('current (nA)', 'FontName', 'helvetica', 'FontSize', fontsize);
ylabel('avg. firing rate', 'FontName', 'helvetica', 'FontSize', fontsize);
set(gca,'FontSize', fontsize);
saveas(fig, task + "_ficurveclusters.svg",'svg');
close(fig);

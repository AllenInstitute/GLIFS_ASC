clear all
fig = figure
fig.Renderer='Painters';

task = "smnist"; % smnist or pattern
fontsize = 24;

% MEM: thresh, k_m
% ASC: k, r, amp

specname = "2-final-256units";
% specname = "4/pattern-4-128units";
if strcmp(task, "smnist")
    mem_params = xlsread("results_wkof_080821/smnist-" + specname + "-0itr-membraneparams.csv");
    asc_params = xlsread("results_wkof_080821/smnist-" + specname + "-0itr-ascparams.csv");
    
    mem_params_init = xlsread("results_wkof_080821/smnist-" + specname + "-0itr-init-membraneparams.csv");
    asc_params_init = xlsread("results_wkof_080821/smnist-" + specname + "-0itr-init-ascparams.csv");
else
    mem_params = xlsread("paper_results/pattern_results/pattern-" + specname + "-0itr-membraneparams.csv");
    asc_params = xlsread("paper_results/pattern_results/pattern-" + specname + "-0itr-ascparams.csv");
    
    mem_params_init = xlsread("paper_results/pattern_results/pattern-" + specname + "-0itr-init-membraneparams.csv");
    asc_params_init = xlsread("paper_results/pattern_results/pattern-" + specname + "-0itr-init-ascparams.csv");
end

nbins = 20;

% h(1) = subplot(3,2,1);
subplot(5, 1, 1);
hh = histogram(mem_params(:,1), nbins, 'FaceAlpha', 1, 'FaceColor', "#332288");
hold on
histogram(mem_params_init(:,1), hh.BinEdges, 'FaceAlpha', 0.8, 'FaceColor', '#117733');
xlabel("thresh (mV)", 'FontSize', fontsize);
% set(gca,'FontSize', fontsize-8);

% fig = figure
% fig.Renderer='Painters';
% % h(2) = subplot(3,2,2);
subplot(5, 1, 2);
hh = histogram(mem_params(:,2), nbins, 'FaceAlpha', 1, 'FaceColor', "#332288");
hold on
histogram(mem_params_init(:,2), hh.BinEdges, 'FaceAlpha', 0.8, 'FaceColor', '#117733');
xlabel("k_m (1/ms)", 'FontSize', fontsize);
% set(gca,'FontSize', fontsize-8);

% fig = figure
% fig.Renderer='Painters';
% h(3) = subplot(3,2,3);
subplot(5, 1, 3);
hh = histogram(asc_params(:,1), nbins, 'FaceAlpha', 1, 'FaceColor', "#332288");
hold on
histogram(asc_params_init(:,1), hh.BinEdges, 'FaceAlpha', 0.8, 'FaceColor', "#117733");
xlabel("k_j (1/ms)", 'FontSize', fontsize);
% set(gca,'FontSize', fontsize-8);

% fig = figure
% fig.Renderer='Painters';
subplot(5, 1, 4);
% h(4) = subplot(3,2,4);
hh = histogram(asc_params(:,2), nbins, 'FaceAlpha', 1, 'FaceColor', "#332288");
hold on
histogram(asc_params_init(:,2), hh.BinEdges, 'FaceAlpha', 0.8, 'FaceColor', '#117733');
xlabel("r_j", 'FontSize', fontsize);
% set(gca,'FontSize', fontsize-8);

% fig = figure
% fig.Renderer='Painters';
subplot(5, 1, 5);
% h(5) = subplot(3,2,5); % the last (odd) axes
hh = histogram(asc_params(:,3), nbins, 'FaceAlpha', 1, 'FaceColor', "#332288");
hold on
histogram(asc_params_init(:,3), hh.BinEdges, 'FaceAlpha', 0.8, 'FaceColor', '#117733');
xlabel("a_j (pA)", 'FontSize', fontsize);
% set(gca,'FontSize', fontsize-8);

% pos = get(h,'Position');
% new = mean(cellfun(@(v)v(1),pos(1:2)));
% set(h(5),'Position',[new,pos{end}(2:end)])
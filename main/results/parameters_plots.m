clear all
fig = figure
fig.Renderer='Painters';

task = "smnist"; % smnist or pattern
fontsize = 24;

fincolor = "#40B0A6";
initcolor = "#E1BE6A";

% MEM: thresh, k_m
% ASC: k, r, amp

specname = "2-final-256units";
num = 2;
% specname = "2/pattern-2-128units";
if strcmp(task, "smnist")
    mem_params = xlsread("paper_results/smnist_results/smnist-" + string(num) + "/smnist-" + specname + "-0itr-membraneparams.csv");
    asc_params = xlsread("paper_results/smnist_results/smnist-" + string(num) + "/smnist-" + specname + "-0itr-ascparams.csv");
    
    mem_params_init = xlsread("paper_results/smnist_results/smnist-" + string(num) + "/smnist-" + specname + "-0itr-init-membraneparams.csv");
    asc_params_init = xlsread("paper_results/smnist_results/smnist-" + string(num) + "/smnist-" + specname + "-0itr-init-ascparams.csv");
else
    mem_params = xlsread("paper_results/pattern_results/pattern-" + specname + "-0itr-membraneparams.csv");
    asc_params = xlsread("paper_results/pattern_results/pattern-" + specname + "-0itr-ascparams.csv");
    
    mem_params_init = xlsread("paper_results/pattern_results/pattern-" + specname + "-0itr-init-membraneparams.csv");
    asc_params_init = xlsread("paper_results/pattern_results/pattern-" + specname + "-0itr-init-ascparams.csv");
end

nbins = 20;

h(1) = subplot(3,2,1);
% subplot(3, 2, 1);
hh = histogram(mem_params(:,1), nbins, 'FaceAlpha', 1, 'FaceColor', fincolor);
hold on
histogram(mem_params_init(:,1), hh.BinEdges, 'FaceAlpha', 0.8, 'FaceColor', initcolor);
Ylm = ylim;
ylim([Ylm(1), 1.1 * Ylm(2)]);
xlabel("thresh (mV)", 'FontSize', fontsize);
ylabel("count", 'FontSize', fontsize);
set(gca,'FontSize', fontsize-8);

% fig = figure
% fig.Renderer='Painters';
h(2) = subplot(3,2,2);
% subplot(3, 2, 2);
hh = histogram(mem_params(:,2), nbins, 'FaceAlpha', 1, 'FaceColor', fincolor);
hold on
histogram(mem_params_init(:,2), hh.BinEdges, 'FaceAlpha', 0.8, 'FaceColor', initcolor);
Ylm = ylim;
ylim([Ylm(1), 1.1 * Ylm(2)]);
ylabel("count", 'FontSize', fontsize);
xlabel("k_m (1/ms)", 'FontSize', fontsize);
set(gca,'FontSize', fontsize-8);

% fig = figure
% fig.Renderer='Painters';
h(3) = subplot(3,2,3);
% subplot(3, 2, 3);
hh = histogram(asc_params(:,1), nbins, 'FaceAlpha', 1, 'FaceColor', fincolor);
hold on
histogram(asc_params_init(:,1), hh.BinEdges, 'FaceAlpha', 0.8, 'FaceColor', initcolor);
Ylm = ylim;
ylim([Ylm(1), 1.1 * Ylm(2)]);
xlabel("k_j (1/ms)", 'FontSize', fontsize);
ylabel("count", 'FontSize', fontsize);
set(gca,'FontSize', fontsize-8);

% fig = figure
% fig.Renderer='Painters';
% subplot(3, 2, 4);
h(4) = subplot(3,2,4);
hh = histogram(asc_params(:,2), nbins, 'FaceAlpha', 1, 'FaceColor', fincolor);
hold on
histogram(asc_params_init(:,2), hh.BinEdges, 'FaceAlpha', 0.8, 'FaceColor', initcolor);
Ylm = ylim;
ylim([Ylm(1), 1.1 * Ylm(2)]);
xlabel("r_j", 'FontSize', fontsize);
ylabel("count", 'FontSize', fontsize);
set(gca,'FontSize', fontsize-8);

% fig = figure
% fig.Renderer='Painters';
% subplot(3, 2, 5);
h(5) = subplot(3,2,5); % the last (odd) axes
hh = histogram(asc_params(:,3), nbins, 'FaceAlpha', 1, 'FaceColor', fincolor);
hold on
histogram(asc_params_init(:,3), hh.BinEdges, 'FaceAlpha', 0.8, 'FaceColor', initcolor);
Ylm = ylim;
ylim([Ylm(1), 1.1 * Ylm(2)]);
xlabel("a_j (pA)", 'FontSize', fontsize);
ylabel("count", 'FontSize', fontsize);
set(gca,'FontSize', fontsize-8);


pos = get(h,'Position');
new = mean(cellfun(@(v)v(1),pos(1:2)));
set(h(5),'Position',[new,pos{end}(2:end)])

% pos = get(h,'Position');
% new = mean(cellfun(@(v)v(1),pos(1:2)));
% set(h(5),'Position',[new,pos{end}(2:end)])
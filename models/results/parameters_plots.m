fig = figure
fig.Renderer='Painters';

task = "smnist"; % smnist or pattern

% MEM: thresh, k_m
% ASC: k, r, amp

if strcmp(task, "smnist")
    mem_params = xlsread("results_wkof_080121/smnist-rglif-2asc-256units-0itr-membraneparams.csv");
    asc_params = xlsread("results_wkof_080121/smnist-rglif-2asc-256units-0itr-ascparams.csv");
else
    mem_params = xlsread("results_wkof_080121/pattern-rglif-2asc-128units-0itr-membraneparams.csv");
    asc_params = xlsread("results_wkof_080121/pattern-rglif-2asc-128units-0itr-ascparams.csv");
end

nbins = 20;

h(1) = subplot(3,2,1);
histogram(mem_params(:,1), nbins);
xlabel("threshold (mV)", 'FontSize', 12);

h(2) = subplot(3,2,2);
histogram(1 ./ mem_params(:,2), nbins);
xlabel("membrane tau (ms)", 'FontSize', 12);

h(3) = subplot(3,2,3);
histogram(1 ./ asc_params(:,1), nbins);
xlabel("ASC tau (ms)", 'FontSize', 12);

h(4) = subplot(3,2,4);
histogram(asc_params(:,2), nbins);
xlabel("ASC multiplicative", 'FontSize', 12);

h(5) = subplot(3,2,5); % the last (odd) axes
histogram(asc_params(:,3), nbins);
xlabel("ASC additive", 'FontSize', 12);

pos = get(h,'Position');
new = mean(cellfun(@(v)v(1),pos(1:2)));
set(h(5),'Position',[new,pos{end}(2:end)])
clear all
fig = figure
fig.Renderer='Painters';

task = "smnist"; % smnist or pattern
fontsize = 24;

% MEM: thresh, k_m
% ASC: k, r, amp

specname = "4-wreg-256units";
if strcmp(task, "smnist")
    mem_params = xlsread("results_wkof_080821/smnist-" + specname + "-0itr-membraneparams.csv");
    asc_params = xlsread("results_wkof_080821/smnist-" + specname + "-0itr-ascparams.csv");
    
    mem_params_init = xlsread("results_wkof_080821/smnist-" + specname + "-0itr-init-membraneparams.csv");
    asc_params_init = xlsread("results_wkof_080821/smnist-" + specname + "-0itr-init-ascparams.csv");
else
    mem_params = xlsread("results_wkof_080121/pattern-" + str(num) + "-128units-0itr-membraneparams.csv");
    asc_params = xlsread("results_wkof_080121/pattern-" + str(num) + "-128units-0itr-ascparams.csv");
    
    mem_params_init = xlsread("results_wkof_080121/pattern-" + str(num) + "-128units-0itr-init-membraneparams.csv");
    asc_params_init = xlsread("results_wkof_080121/pattern-" + str(num) + "-128units-0itr-init-ascparams.csv");
end

nbins = 20;

h(1) = subplot(3,2,1);
hh = histogram(mem_params(:,1), nbins);
hold on
histogram(mem_params_init(:,1), hh.BinEdges);
xlabel("threshold (mV)", 'FontSize', fontsize);

h(2) = subplot(3,2,2);
hh = histogram(1 ./ mem_params(:,2), nbins);
hold on
histogram(1 ./ mem_params_init(:,2), hh.BinEdges);
xlabel("membrane tau (ms)", 'FontSize', fontsize);

h(3) = subplot(3,2,3);
hh = histogram(1 ./ asc_params(:,1), nbins);
hold on
histogram(1 ./ asc_params_init(:,1), hh.BinEdges);
xlabel("ASC tau (ms)", 'FontSize', fontsize);

h(4) = subplot(3,2,4);
hh = histogram(asc_params(:,2), nbins);
hold on
histogram(asc_params_init(:,2), hh.BinEdges);
xlabel("ASC multiplicative", 'FontSize', fontsize);

h(5) = subplot(3,2,5); % the last (odd) axes
hh = histogram(asc_params(:,3), nbins);
hold on
histogram(asc_params_init(:,3), hh.BinEdges);
xlabel("ASC additive (pA)", 'FontSize', fontsize);

pos = get(h,'Position');
new = mean(cellfun(@(v)v(1),pos(1:2)));
set(h(5),'Position',[new,pos{end}(2:end)])
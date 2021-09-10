fig = figure
fig.Renderer='Painters';
dt = 0.05

task = "smnist"; % smnist or pattern
fontsize = 24;
linewidth = 2;

% MEM: thresh, k_m
% ASC: k, r, amp

name = "4-agn";
clustername = "smnist-4-agn-256units-256units-0itr-allparams-clusters";
paramname = "smnist-4-agn-256units-256units-0itr-allparams";
initparamname = "smnist-4-agn-256units-init-256units-0itr-allparams";
if strcmp(task, "smnist")
%     slopes = xlsread("results_wkof_080821/smnist-" + name + "-ficurve-slopes.csv");
    i_syns = xlsread("results_wkof_080821/smnist-" + name + "-ficurve-isyns.csv");
    f_rates = xlsread("results_wkof_080821/smnist-" + name + "-ficurve-frates.csv");
    
    clusters = xlsread(strcat("results_wkof_080821/", clustername, ".csv"));
    parameters = xlsread(strcat("results_wkof_080821/", paramname, ".csv"));
    initparameters = xlsread(strcat("results_wkof_080821/", initparamname, ".csv"));
    
%     slopes_init = xlsread("results_wkof_080821/smnist-" + name + "-init-ficurve-slopes.csv");
    i_syns_init = xlsread("results_wkof_080821/smnist-" + name + "-init-ficurve-isyns.csv");
    f_rates_init = xlsread("results_wkof_080821/smnist-" + name + "-init-ficurve-frates.csv");
elseif strcmp(task, "pattern")
    i_syns = xlsread("results_wkof_080821/pattern-" + name + "-ficurve-isyns.csv");
    f_rates = xlsread("results_wkof_080821/pattern-" + name + "-ficurve-frates.csv");
    
    i_syns_init = xlsread("results_wkof_080821/pattern-" + name + "-init-ficurve-isyns.csv");
    f_rates_init = xlsread("results_wkof_080821/pattern-" + name + "-init-ficurve-frates.csv");
elseif strcmp(task, "smnist-wtonly")
    slopes = xlsread("results_wkof_080121/smnist-wtonly-rglif-ficurve-slopes.csv");
    i_syns = xlsread("results_wkof_080121/smnist-wtonly-rglif-ficurve-isyns.csv");
    f_rates = xlsread("results_wkof_080121/smnist-wtonly-rglif-ficurve-frates.csv");
elseif strcmp(task, "pattern-wtonly")
    slopes = xlsread("results_wkof_080121/pattern-wtonly-rglif-ficurve-slopes.csv");
    i_syns = xlsread("results_wkof_080121/pattern-wtonly-rglif-ficurve-isyns.csv");
    f_rates = xlsread("results_wkof_080121/pattern-wtonly-rglif-ficurve-frates.csv");
end

colors = ["#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677", "#AA4499", "#882255", "#72B803", "#109EC4", "#4DB8F6", "#4E1D87"];

for i=1:size(f_rates, 1)
    plot(i_syns, f_rates(:,i), 'LineWidth', linewidth, 'color', colors(clusters(i)+1))
    hold on
end

for i=1:size(f_rates_init, 1)
    plot(i_syns_init, f_rates_init(:,i), 'k', 'LineWidth', linewidth)
    hold on
end


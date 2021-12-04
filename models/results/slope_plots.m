fig = figure
fig.Renderer='Painters';
dt = 0.05

task = "pattern"; % smnist or pattern
fontsize = 24;
linewidth = 2;

% MEM: thresh, k_m
% ASC: k, r, amp

clustername = "smnist-4-final-256units-0itr-allparams-clusters";
paramname = "smnist-4-final-256units-0itr-allparams";
initparamname = "smnist-4-final-256units-0itr-init-allparams";

if strcmp(task, "smnist")
   name = "4-final";

%     slopes = xlsread("results_wkof_080821/smnist-" + name + "-ficurve-slopes.csv");
    i_syns = (1 / 1000) * xlsread("results_wkof_080821/smnist-" + name + "-isyns.csv");
    f_rates = xlsread("results_wkof_080821/smnist-" + name + "-frates.csv");
    
    clusters = xlsread(strcat("results_wkof_080821/", clustername, ".csv"));
    parameters = xlsread(strcat("results_wkof_080821/", paramname, ".csv"));
    initparameters = xlsread(strcat("results_wkof_080821/", initparamname, ".csv"));
    
%     slopes_init = xlsread("results_wkof_080821/smnist-" + name + "-init-ficurve-slopes.csv");
    i_syns_init = (1 / 1000) * xlsread("results_wkof_080821/smnist-" + name + "-init-isyns.csv");
    f_rates_init = xlsread("results_wkof_080821/smnist-" + name + "-init-frates.csv");
elseif strcmp(task, "pattern")
    name = "4";

    i_syns = (1/1000) * xlsread("paper_results/pattern_results/pattern-4/pattern-" + name + "-ficurve-isyns.csv");
    f_rates = xlsread("paper_results/pattern_results/pattern-4/pattern-" + name + "-ficurve-frates.csv");
    
    i_syns_init = (1/1000) * xlsread("paper_results/pattern_results/pattern-4/pattern-" + name + "-init-ficurve-isyns.csv");
    f_rates_init = xlsread("paper_results/pattern_results/pattern-4/pattern-" + name + "-init-ficurve-frates.csv");
elseif strcmp(task, "smnist-wtonly")
    slopes = xlsread("results_wkof_080121/smnist-wtonly-rglif-ficurve-slopes.csv");
    i_syns = xlsread("results_wkof_080121/smnist-wtonly-rglif-ficurve-isyns.csv");
    f_rates = xlsread("results_wkof_080121/smnist-wtonly-rglif-ficurve-frates.csv");
elseif strcmp(task, "pattern-wtonly")
    slopes = xlsread("results_wkof_080121/pattern-wtonly-rglif-ficurve-slopes.csv");
    i_syns = xlsread("results_wkof_080121/pattern-wtonly-rglif-ficurve-isyns.csv");
    f_rates = xlsread("results_wkof_080121/pattern-wtonly-rglif-ficurve-frates.csv");
end

colors = ["#332288", "#44AA99", "#DDCC77", "#CC6677", "#AA4499", "#882255", "#72B803", "#109EC4", "#4DB8F6", "#4E1D87"];

for i=1:size(f_rates, 1)
    if strcmp(task, "smnist")
        plot(i_syns, f_rates(:,i), 'LineWidth', linewidth, 'color', colors(clusters(i)+1))
    else
        plot(i_syns, f_rates(:,i), 'LineWidth', linewidth)
    end
    hold on
end

for i=1:size(f_rates_init, 1)
    plot(i_syns_init, f_rates_init(:,i), 'k', 'LineWidth', linewidth)
    hold on
end
set(gca,'FontSize', fontsize);
xlabel('current (nA)', 'FontName', 'helvetica', 'FontSize', fontsize);
ylabel('avg. firing rate', 'FontName', 'helvetica', 'FontSize', fontsize);
set(gca,'FontSize', fontsize);

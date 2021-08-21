fig = figure
fig.Renderer='Painters';
dt = 0.05

task = "pattern"; % smnist or pattern
fontsize = 24;
linewidth = 2;

% MEM: thresh, k_m
% ASC: k, r, amp

if strcmp(task, "smnist")
    slopes = xlsread("results_wkof_080121/smnist-rglif-2asc-wreg-ficurve-slopes.csv");
    i_syns = xlsread("results_wkof_080121/smnist-rglif-2asc-wreg-ficurve-isyns.csv");
    f_rates = xlsread("results_wkof_080121/smnist-rglif-2asc-wreg-ficurve-frates.csv");
elseif strcmp(task, "pattern")
    slopes = xlsread("results_wkof_080121/pattern-rglif-wreg-ficurve-slopes.csv");
    i_syns = xlsread("results_wkof_080121/pattern-rglif-wreg-ficurve-isyns.csv");
    f_rates = xlsread("results_wkof_080121/pattern-rglif-wreg-ficurve-frates.csv");
elseif strcmp(task, "smnist-wtonly")
    slopes = xlsread("results_wkof_080121/smnist-wtonly-rglif-ficurve-slopes.csv");
    i_syns = xlsread("results_wkof_080121/smnist-wtonly-rglif-ficurve-isyns.csv");
    f_rates = xlsread("results_wkof_080121/smnist-wtonly-rglif-ficurve-frates.csv");
elseif strcmp(task, "pattern-wtonly")
    slopes = xlsread("results_wkof_080121/pattern-wtonly-rglif-ficurve-slopes.csv");
    i_syns = xlsread("results_wkof_080121/pattern-wtonly-rglif-ficurve-isyns.csv");
    f_rates = xlsread("results_wkof_080121/pattern-wtonly-rglif-ficurve-frates.csv");
end

for i=1:size(f_rates, 1)
    plot(i_syns, f_rates(:,i), 'LineWidth', linewidth)
    hold on
end

xlim([-5000, 5000]);
xlabel('current (pA)', 'FontName', 'helvetica', 'FontSize', fontsize);
ylabel('avg. firing probability', 'FontName', 'helvetica', 'FontSize', fontsize);
set(gca,'FontSize', fontsize);
% nbins = 20;
% 
% histogram(slopes, nbins);
% xlabel("f/I slope", 'FontSize', fontsize);
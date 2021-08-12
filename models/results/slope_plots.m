fig = figure
fig.Renderer='Painters';

task = "pattern"; % smnist or pattern
fontsize = 24;

% MEM: thresh, k_m
% ASC: k, r, amp

if strcmp(task, "smnist")
    slopes = xlsread("results_wkof_080121/smnist-2asc-rglif-ficurve-slopes.csv");
else
    slopes = xlsread("results_wkof_080121/pattern-2asc-rglif-ficurve-slopes.csv");
end

nbins = 20;

histogram(slopes, nbins);
xlabel("f/I slope", 'FontSize', fontsize);
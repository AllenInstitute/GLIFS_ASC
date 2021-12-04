clear all
targets = xlsread("paper_results/pattern_results/pattern-9/pattern-9-131units-5itr-targets.csv");

order = uint8([9, 10, 3, 7, 1, 5, 2, 6, 4, 8]);
params = uint8([131, 64, 131, 131, 131, 131, 128, 130, 128, 130]);

linewidth = 2;
fontsize=24;

sim_time = 5;
colors = ["#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677", "#AA4499", "#882255", "#72B803", "#109EC4", "#4DB8F6", "#4E1D87"];

fig =figure;%('Position', [0,0,400,250]);
fig.Renderer='Painters';

for i = [1:6]%[1:size(traces,2)]
    subplot(3, 2, i)
    plot([0:0.05:sim_time-0.05], targets(:, i), 'k', 'LineWidth', linewidth, 'HandleVisibility', 'off');
    hold on
    handle_vis = 'on';
    if i > 1
        handle_vis = 'off';
    end
    for j = [1:10]
        trace = xlsread("paper_results/pattern_results/pattern-" + string(order(j)) + "/pattern-" + string(order(j)) + "-" + string(params(j)) + "units-5itr-finaloutputs.csv");
        plot([0:0.05:sim_time-0.05], trace(:, i), 'Color', colors(j), 'LineWidth', linewidth, 'HandleVisibility', handle_vis);
        hold on
    end
end

ylabel("network output", 'FontName', 'helvetica', 'FontSize', fontsize);
ylabel("time (ms)", 'FontName', 'helvetica', 'FontSize', fontsize);

legend("RNN", "LSTM", "HomA", "Hom", "FHetA", "FHet", "RHetA", "RHet", "LHetA", "LHet", 'FontSize', fontsize);
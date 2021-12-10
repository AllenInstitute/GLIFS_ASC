%% Produces Figure 4D

clear all
targets = xlsread("paper_results/pattern_results/final_outputs/pattern-9-5itr-targets.csv");

order = uint8([9, 10, 3, 7, 1, 5, 2, 6, 4, 8]);
nums = uint8([5, 5, 0, 5, 5, 5, 0, 5, 5, 5]);
labels = {"RNN", "LSTM", "HomA", "Hom", "FHetA", "FHet", "RHetA", "RHet", "LHetA", "LHet"}
linewidth = 2;
fontsize=24;

sim_time = 5;
colors = ["#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677", "#AA4499", "#882255", "#72B803", "#109EC4", "#4DB8F6", "#4E1D87"];

fig =figure;
fig.Renderer='Painters';

idx = [1,2,4,5,7,8];
for i = [1:6]
    subplot(3, 3, idx(i))
    ps = [];
    for j = [1:10]
        trace = xlsread("paper_results/pattern_results/final_outputs/pattern-" + string(order(j)) + "-" + string(nums(j)) + "itr-learnedoutputs.csv");
        ps(end + 1) = plot([0:0.05:sim_time-0.05], trace(:, i), 'Color', colors(j), 'LineWidth', linewidth, 'HandleVisibility', 'on', 'DisplayName', labels{j});
        hold on
    end
    plot([0:0.05:sim_time-0.05], targets(:, i), 'k', 'LineWidth', linewidth, 'HandleVisibility', 'off');
    xlim([0,5]);
    set(gca,'Xtick',0:1:5);
    if i > 4
        xlabel("time (ms)", 'FontName', 'helvetica', 'FontSize', fontsize);
    end
    set(gca,'FontName', 'helvetica', 'FontSize', fontsize);
    if i == 4
        subplot(3,3,[3,6,9]);
        axis off
        legend(ps, "RNN", "LSTM", "HomA", "Hom", "FHetA", "FHet", "RHetA", "RHet", "LHetA", "LHet", 'FontSize', fontsize);
    end
end
han=axes(fig,'visible','off'); 
han.Title.Visible='on';
han.XLabel.Visible='on';
han.YLabel.Visible='on';

ylabel(han, "network output", 'FontName', 'helvetica', 'FontSize', fontsize);
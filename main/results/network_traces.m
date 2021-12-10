filename = "results_wkof_080821/smnist-4-anneal-sampleresponses.csv"
traces = xlsread(filename)

linewidth = 2;
fontsize=24;

sim_time = 5;
colors = ["#332288", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677", "#AA4499", "#882255"];

fig =figure;%('Position', [0,0,400,250]);
fig.Renderer='Painters';
for i = [1:6]%[1:size(traces,2)]
    subplot(3, 2, i)
    trace = traces(:, randi(256))
    plot([0:0.05:sim_time-0.05], trace, 'LineWidth', linewidth);
    hold on
end

%legend('FontName', 'helvetica', 'FontSize', fontsize, 'Location', 'best')

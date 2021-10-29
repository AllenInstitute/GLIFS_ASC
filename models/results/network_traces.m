filename = "results_wkof_080821/smnist-4-anneal-sampleresponses.csv"
traces = xlsread(filename)

linewidth = 2;
fontsize=24;

sim_time = 100;
colors = ["#332288", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677", "#AA4499", "#882255"];

fig =figure;%('Position', [0,0,400,250]);
fig.Renderer='Painters';
subplot(5,1,1);
for i = [1:size(traces,2)]
    trace = traces(:, i)
    plot([0:0.05:sim_time-0.05], trace, 'LineWidth', linewidth);
    hold on
end
set(gca,'FontSize', fontsize);
ylabel('input', 'FontName', 'helvetica', 'FontSize', fontsize);

%legend('FontName', 'helvetica', 'FontSize', fontsize, 'Location', 'best')
xlabel('time (ms)', 'FontName', 'helvetica', 'FontSize', fontsize);
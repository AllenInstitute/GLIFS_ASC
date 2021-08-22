neg = xlsread("results_wkof_080821/sample-outputs-neg.csv");
zero = xlsread("results_wkof_080821/sample-outputs-zero.csv");
opp = xlsread("results_wkof_080821/sample-outputs-opp.csv");

filenames = ["results_wkof_080821/sample-outputs-neg", "results_wkof_080821/sample-outputs-zero", "results_wkof_080821/sample-outputs-opp"];
lists = [neg, zero, opp];
names = ["A", "B", "C"];

linewidth = 2;
fontsize=24;

sim_time = 40;
colors = ["#332288", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677", "#AA4499", "#882255"];

fig =figure('Position', [0,0,400,250]);
fig.Renderer='Painters';
for i = [1:3]
    ins = xlsread(filenames(i) + "in.csv");
    plot([0:0.05:sim_time-0.05], ins, 'Color', colors(i), 'LineWidth', linewidth, 'DisplayName', names(i));
    hold on
end
set(gca,'FontSize', fontsize);
ylabel('input', 'FontName', 'helvetica', 'FontSize', fontsize);

fig =figure('Position', [0,0,400,250]);
fig.Renderer='Painters';
for i = [1:3]
    syns = xlsread(filenames(i) + "syn.csv");
    plot([0:0.05:sim_time-0.05], syns, 'Color', colors(i), 'LineWidth', linewidth, 'DisplayName', names(i));
    hold on
end
set(gca,'FontSize', fontsize);
ylabel('Isyn (pA)', 'FontName', 'helvetica', 'FontSize', fontsize);

fig =figure('Position', [0,0,400,250]);
fig.Renderer='Painters';
for i = [1:3]
    ascs = xlsread(filenames(i) + "asc.csv");
    plot([0:0.05:sim_time-0.05], ascs, 'Color', colors(i), 'LineWidth', linewidth, 'DisplayName', names(i));
    hold on
end
set(gca,'FontSize', fontsize);
ylabel('ASC (pA)', 'FontName', 'helvetica', 'FontSize', fontsize);

fig =figure('Position', [0,0,400,250]);
fig.Renderer='Painters';
for i = [1:3]
    voltages = xlsread(filenames(i) + "voltage.csv");
    plot([0:0.05:sim_time-0.05], voltages, 'Color', colors(i), 'LineWidth', linewidth, 'DisplayName', names(i));
    hold on
end
set(gca,'FontSize', fontsize);
ylabel('voltage (mV)', 'FontName', 'helvetica', 'FontSize', fontsize);

fig =figure('Position', [0,0,400,250]);
fig.Renderer='Painters';
for i = [1:3]
    firing = xlsread(filenames(i) + ".csv");
    plot([0:0.05:sim_time-0.05], firing, 'Color', colors(i), 'LineWidth', linewidth, 'DisplayName', names(i));
    hold on
end
set(gca,'FontSize', fontsize);
ylabel('firing', 'FontName', 'helvetica', 'FontSize', fontsize);

%legend('FontName', 'helvetica', 'FontSize', fontsize, 'Location', 'best')
xlabel('time (ms)', 'FontName', 'helvetica', 'FontSize', fontsize);
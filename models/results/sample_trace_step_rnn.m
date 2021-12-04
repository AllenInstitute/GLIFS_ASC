%% Produces traces in Figure 1A

clear all;
step = false;

filename = "paper_results/sample_outputs/sample-outputs-step/sample-outputs-step-rnn-sample-outputs-step-rnn"

linewidth = 2;
fontsize=18;
fontsize_label=24;

sim_time = 40;
colors = ["#332288", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677", "#AA4499", "#882255"];

ylabels = ["input", "V(mV)", "firing"];
ylims = {[0,2], [0,2],[-20, 20], [-80, 20], [0, 1.2]};

toplot = {}

toplottemp = cell(0); 
filenamestemp = ["in.csv", "voltage.csv", ".csv"]
for f = 1:length(filenamestemp)
    c = 1;
    toplottemp{f} = c * xlsread(filename + filenamestemp(f));
end
toplot{end+1} = toplottemp;


fig =figure;
fig.Renderer='Painters';
yls = {};
minx = 100;
for i = [1:3]
    subplot(3, 1, i);
    plot([0:0.05:sim_time-0.05], toplot{1}{i}, 'Color', colors(1), 'LineWidth', linewidth);
    hold on

    set(gca,'FontSize', fontsize);
        
    ylim([0, 15]);
    yl = ylabel(ylabels(i), 'FontName', 'helvetica', 'FontSize', fontsize_label);

    set(gca,'XTick',[])
    set(gca,'YTick',[])
end
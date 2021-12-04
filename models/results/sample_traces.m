%% Produces Figure 2A, 2B
clear all;
step = false; % true for figure 2b and false for figure 2a

filenames_lowsigmav = ["paper_results/sample_outputs/sample-outputs/sample-outputs-sigmav1e-3-neg", "paper_results/sample_outputs/sample-outputs/sample-outputs-sigmav1e-3-zero", "paper_results/sample_outputs/sample-outputs/sample-outputs-sigmav1e-3-opp"];
filenames = ["paper_results/sample_outputs/sample-outputs/sample-outputs-neg", "paper_results/sample_outputs/sample-outputs/sample-outputs-zero", "paper_results/sample_outputs/sample-outputs/sample-outputs-opp"];

if step
    filenames_lowsigmav = ["paper_results/sample_outputs/sample-outputs-steps/sample-outputs-steps-sigmav1-3-0", "paper_results/sample_outputs/sample-outputs-steps/sample-outputs-steps-sigmav1-3-1", "paper_results/sample_outputs/sample-outputs-steps/sample-outputs-steps-sigmav1-3-2"];
    filenames = ["paper_results/sample_outputs/sample-outputs-steps/sample-outputs-steps-0", "paper_results/sample_outputs/sample-outputs-steps/sample-outputs-steps-1", "paper_results/sample_outputs/sample-outputs-steps/sample-outputs-steps-2"];
end
names = ["A", "B", "C"];

linewidth = 2;
fontsize=18;
fontsize_label=24;

sim_time = 40;
colors = ["#332288", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677", "#AA4499", "#882255"];

ylabels = ["input", "Is(pA)", "Ia(nA)", "V(mV)", "firing"];
ylims = {[0,2], [0,2],[-20, 20], [-80, 20], [0, 1.2]};
if step
    ylims = {[0,20], [0,20],[-100, 20], [-280, 20], [0, 1.2]};
end
toplot = {}

for j = [1,3]
    toplottemp = cell(0); 
    filenamestemp = ["in.csv", "syn.csv", "asc.csv", "voltage.csv", ".csv"]
    for f = 1:length(filenamestemp)
        c = 1;
        if f == 3
            c = 0.001;
        end
        toplottemp{f} = c * xlsread(filenames(j) + filenamestemp(f));
        toplottemp{f+5} = c * xlsread(filenames_lowsigmav(j) + filenamestemp(f));
    end
    toplot{end+1} = toplottemp;
end

fig =figure;
fig.Renderer='Painters';
for i = [1:10]
    subplot(5, 2, i);
    for j = [1:2]
        plot([0:0.05:sim_time-0.05], toplot{j}{round(floor((i+1) / 2)) + (5 * mod(i-1, 2))}, 'Color', colors(j), 'LineWidth', linewidth, 'DisplayName', names(j));
        hold on
    end
    set(gca,'FontSize', fontsize);
    if mod(i-1, 2) == 0
        ylim(ylims{round(floor((i+1) / 2))});
        Ylm=ylim;                          % get x, y axis limits 
        yl = ylabel(ylabels(round(floor((i+1) / 2))), 'Position', [-8, mean(Ylm)], 'FontName', 'helvetica', 'FontSize', fontsize_label);
    end
    if i > 8
        xlabel('time (ms)', 'FontName', 'helvetica', 'FontSize', fontsize_label);
    else
        set(gca,'xticklabel',{[]})
    end
    if mod(i, 2) == 0
        set(gca,'yticklabel',{[]})
    end
    round(floor((i+1) / 2))
    ylim(ylims{round(floor((i+1) / 2))});\
    yticks(linspace(ylim_temp(1), ylim_temp(2), 5));
end

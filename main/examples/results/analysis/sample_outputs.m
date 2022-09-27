%% Produces Figure 2

clear all;
step = true; % true for figure 2b and false for figure 2a

filenames_lowsigmav = ["./../sample_outputs/sample-outputs/sample-outputs-sigmav1e-3-neg", "./../sample_outputs/sample-outputs/sample-outputs-sigmav1e-3-zero", "./../sample_outputs/sample-outputs/sample-outputs-sigmav1e-3-opp"];
filenames = ["./../sample_outputs/sample-outputs/sample-outputs-neg", "./../sample_outputs/sample-outputs/sample-outputs-zero", "./../sample_outputs/sample-outputs/sample-outputs-opp"];
legend_entries = [strcat("r_{1},r_{2} = -1; a_{1},a_{2} = -5nA"), strcat("r_{1}= -1, r_{2} = 1; a_{1} = 5nA, a_{2} = -5nA")];

if step
    filenames_lowsigmav = ["./../sample_outputs/sample-outputs-steps/sample-outputs-steps-sigmav1-3-0", "./../sample_outputs/sample-outputs-steps/sample-outputs-steps-sigmav1-3-1", "./../sample_outputs/sample-outputs-steps/sample-outputs-steps-sigmav1-3-2"];
    filenames = ["./../sample_outputs/sample-outputs-steps/sample-outputs-steps-0", "./../sample_outputs/sample-outputs-steps/sample-outputs-steps-1", "./../sample_outputs/sample-outputs-steps/sample-outputs-steps-2"];
    legend_entries = ["input = 0", "input = 10"];
end
names = ["A", "B", "C"];

linewidth = 2;
fontsize=15;
fontsize_label=24;

sim_time = 20;%40;
colors = ["#332288", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677", "#AA4499", "#882255"];

ylabels = ["input_{ }", "I_{syn}(pA)", "I_{j}(nA)", "V_{ }(mV)", "S_{ }"];
ylims = {[0,2], [0,2],[-20, 20], [-80, 20], [0, 1.2]};
if step
    ylims = {[0,20], [0,20],[-100, 20], [-280, 20], [0, 1.2]};
end
toplot = {};

for j = [1,3]
    toplottemp = cell(0); 
    filenamestemp = ["in.csv", "syn.csv", "asc.csv", "voltage.csv", ".csv"];
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

fig = figure('Position', get(0, 'Screensize'));
fig.Renderer='Painters';
for i = [1:10]
    subplot(5, 2, i);
    for j = [1:2]
        x = [0:0.05:sim_time-0.05];
        y = toplot{j}{round(floor((i+1) / 2)) + (5 * mod(i-1, 2))};

        if (size(y,1) == 2) % IE ascurrent
            y = y(:, 1:length(x),:);
        else
            y = y(1:length(x));
        end
        p = plot(x, y, 'Color', colors(j), 'LineWidth', linewidth, 'DisplayName', names(j));
        hold on
    end
    set(gca,'FontSize', fontsize);
    if mod(i-1, 2) == 0
        ylim(ylims{round(floor((i+1) / 2))});
        Ylm=ylim;                          % get x, y axis limits 
        yl = ylabel(ylabels(round(floor((i+1) / 2))), 'Position', [-4, mean(Ylm)], 'FontName', 'helvetica', 'FontSize', fontsize_label);
    end
    if i > 8
        xlabel('time (ms)', 'FontName', 'helvetica', 'FontSize', fontsize_label);
    else
        set(gca,'xticklabel',{[]});
    end
    if mod(i, 2) == 0
        set(gca,'yticklabel',{[]})
    end
    round(floor((i+1) / 2));
    ylim(ylims{round(floor((i+1) / 2))});
    ylim_temp = ylim;
    yticks(linspace(ylim_temp(1), ylim_temp(2), 5));
end
saveas(fig, step + "_sample_output.svg",'svg');
close(fig);

fig = figure('Position', get(0, 'Screensize'));
fig.Renderer='Painters';
% subplot(6,2,1);
plot(nan, nan, 'Color', colors(1), 'LineWidth', linewidth, 'DisplayName', legend_entries(1));
hold on;
plot(nan, nan, 'Color', colors(2), 'LineWidth', linewidth, 'DisplayName', legend_entries(2));
set(gca,'FontSize', fontsize, 'FontName', 'helvetica');
legend('Location', 'southoutside');
saveas(fig, step + "_sample_output_legend.svg",'svg');
close(fig);
% pos = get(h,'Position');
% new = mean(cellfun(@(v)v(1),pos(1:2)));
% set(h(11),'Position',[new,pos{end}(2:end)])

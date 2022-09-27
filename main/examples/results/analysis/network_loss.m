%% Produces Table 2, Figure S1
% t-tests stored to stats_{task}-overalllosses.txt
% figure stored to {task}_errorbars.svg
clearvars;

% TODO: Chnage below three lines
task = "pmnist"; % sine, lmnist, nmnist, pmnist
metric = "acc"; % acc or loss
ylabel_text = "accuracy"; % MSE or accuracy

fontsize = 24;

inset = 1;

results_file = fopen("stats_" + task + "-overalllosses.txt", 'w');

network_types = ["rnn", "lstmn", "glifr_hom", "glifr_homa", "glifr_lhet", "glifr_lheta", "glifr_fhet", "glifr_fheta", "glifr_rhet", "glifr_rheta"];
network_names = ["RNN", "LSTM", "Hom", "HomA", "LHet", "LHetA", "FHet", "FHetA", "RHet", "RHetA"];

if strcmp(task, "pmnist")
    network_types = ["rnn", "glifr_lheta", "glifr_rheta"];
    network_names = ["RNN", "LHetA", "RHetA"];
elseif strcmp(task, "lmnist-anneal")
    network_types = ["glifr_lheta", "glifr_rheta"];
end
test_accs = [];

for n = network_types
    d = dir("./../" + task + "/" + n + "/*/*/*/test_" + metric + ".csv")
    test_accs_n = [];
    for didx = 1:size(d,1)
        filename = d(didx).folder + "\" + d(didx).name;
        test_accs_n = [test_accs_n readmatrix(filename)];
    end
    test_accs = [test_accs; test_accs_n];
end

means = mean(test_accs, 2);
stds = std(test_accs, 0, 2);
for n1 = network_types
    i = find(strcmp(network_types, n1));
    fprintf(results_file, strcat(n1 +  ": ", sprintf('%e', means(i)), " (", sprintf("%9.6f", stds(i)), ")"));
    fprintf(results_file, "\n");
end

fprintf(results_file, "\n");

for n1 = network_types
    for n2 = network_types
        if strcmp(n1, n2)
            continue;
        end
        i1 = find(strcmp(network_types, n1));
        i2 = find(strcmp(network_types, n2));
        % do t-test between them
        [h,p,ci,stats] = ttest2(test_accs(i1,:), test_accs(i2,:));
        fprintf(results_file, strcat(n1 + " vs " + n2 + ": t(", sprintf("%9.6f", stats.df), ") =", sprintf("%9.6f", stats.tstat), " p = ", sprintf('%e', p)));
        fprintf(results_file, '\n');
    end
end
fclose(results_file);

% PLOT DATA
fig = figure('Position', get(0, 'Screensize'));
fig.Renderer='Painters';

fontsize = 24;
alpha = 0.05;

xticks = [1];

means = mean(test_accs, 2);
stds = std(test_accs, 0, 2);
b = bar(xticks, means, 1);

hold on

ngroups = 1;%size(test_accs, 1);
nbars = size(test_accs, 1);
groupwidth = min(0.8, nbars/(nbars + 1.5));

for i = 1:nbars
    x = xticks - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
    errorbar(x, means(i), stds(i), 'k', 'LineWidth', 1, 'CapSize', 10, 'linestyle', 'none', 'HandleVisibility','off');
    hold on
    for j = 1:size(test_accs, 2)
        plot(x,test_accs(i, j),'ko');
        hold on
    end
end

if strcmp(task, "lmnist")
    ylim([0.8, 1]);
elseif strcmp(task, "nmnist")
    ylim([0.8, 0.9]);
end

colors = ["#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677", "#AA4499", "#882255", "#72B803", "#109EC4", "#4DB8F6", "#4E1D87"];

for i = 1:nbars
    b(i).FaceColor = colors(i);
end

set(gca,'xticklabel',{[]})
set(gca, 'FontSize', fontsize);
% set(gca,'XTick', 1:ngroups, 'xticklabel',network_types, 'FontName', 'helvetica', 'FontSize', fontsize);
legend(network_names, 'Location', 'eastoutside', 'FontSize', fontsize);
% xlabel("ratio silenced", 'FontSize', fontsize);
ylabel(ylabel_text, 'FontSize', fontsize);
saveas(fig, task + "_errorbars.svg",'svg');
close(fig);

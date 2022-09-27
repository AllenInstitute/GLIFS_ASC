%% Produces Figure 7 and Figure S3
% t-tests stored to stats_{task}_ablation.txt
clear all

fig = figure('Position', get(0, 'Screensize'));
fig.Renderer='Painters';

fontsize = 24;
alpha = 0.05;

% TODO: modify the below three lines
task = "lmnist"; % sine, lmnist, nmnist, pmnist, lmnist-dropout
metric = "accs"; % accs or losses (if sine)
ylabel_text = "MSE"; % accuracy or MSE (if sine)

results_file = fopen("stats_" + task + "_ablation.txt", 'w');

network_types = ["rnn", "lstmn", "glifr_hom", "glifr_homa", "glifr_lhet", "glifr_lheta", "glifr_fhet", "glifr_fheta", "glifr_rhet", "glifr_rheta"];
network_names = ["RNN", "LSTM", "Hom", "HomA", "LHet", "LHetA", "FHet", "FHetA", "RHet", "RHetA"];

if strcmp(task, "pmnist")
    network_types = ["rnn", "glifr_lheta", "glifr_rheta"];
    network_names = ["RNN", "LHetA", "RHetA"];
end

means = [];
stds = [];
for n = network_types
    if strcmp(task, "lmnist-dropout")
        d = dir("./../" + task + "/" + n + "/ablation_" + metric + ".csv")
    else
        d = dir("./../" + task + "-post/" + n + "/*/*/*/ablation_" + metric + ".csv")
    end

    filename = d(1).folder + "\" + d(1).name;
    ablation_accs_n = readmatrix(filename);
    
    all_results(:, :, find(strcmp(network_types, n))) = ablation_accs_n;
    
    means = [means; mean(ablation_accs_n, 2).'];
    stds = [stds; std(ablation_accs_n, 0, 2).'];        
end

% means = means.';

offset = 0.01;

silence_props = linspace(0,1,6);

% RUN T-TESTS
for i = 1:size(all_results, 1)
    fprintf(results_file, strcat("On percent ", sprintf("%9.6f", silence_props(i)), "\n"));
    for n1 = network_types
        for n2 = network_types
            if ~strcmp(n1, n2)
                i1 = find(strcmp(network_types, n1));
                i2 = find(strcmp(network_types, n2));
                [h, p] = ttest2(all_results(i, :, i1),all_results(i, :, i2));
                fprintf(results_file, strcat(n1, "-", n2, ": ", sprintf("%e", p), "\n"));
            end
        end
    end
end
fclose(results_file);

% PLOT DATA
xticks = 1:6;

b = bar(xticks, means, 1);

hold on

ngroups = size(ablation_accs_n, 1);
nbars = size(means, 1);
groupwidth = min(0.8, nbars/(nbars + 1.5));

for i = 1:nbars
    x = xticks - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
    errorbar(x, means(i,:), stds(i,:), 'k', 'LineWidth', 1, 'CapSize', 10, 'linestyle', 'none', 'HandleVisibility','off');
    hold on
end

colors = ["#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677", "#AA4499", "#882255", "#72B803", "#109EC4", "#4DB8F6", "#4E1D87"];

for i = 1:nbars
    b(i).FaceColor = colors(i);
end

set(gca,'XTick', 1:ngroups, 'xticklabel',silence_props, 'FontName', 'helvetica', 'FontSize', fontsize);
legend(network_names, 'FontSize', fontsize); % Add 'Location', 'west' for sine
xlabel("ratio silenced", 'FontSize', fontsize);
ylabel(ylabel_text);
saveas(fig, task + "_ablation.svg",'svg');
close(fig);
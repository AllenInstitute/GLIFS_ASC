%% Produces Figures 4F, 5D
clearvars;

fig = figure
fig.Renderer='Painters';

task = "pattern"; % smnist or pattern
fontsize = 24;

inset = 1;

if strcmp(task, "smnist")
    results_file = fopen("stats_smnist-overalllosses.txt", 'w');
    data_1 = xlsread("paper_results/smnist_results/smnist-1/smnist-1-final-259units-accs.csv");
    data_2 = xlsread("paper_results/smnist_results/smnist-2/smnist-2-final-256units-accs.csv");
    data_3 = xlsread("paper_results/smnist_results/smnist-3/smnist-3-final-259units-accs.csv");
    data_4 = xlsread("paper_results/smnist_results/smnist-4/smnist-4-final-256units-accs.csv");
    data_5 = xlsread("paper_results/smnist_results/smnist-5/smnist-5-final-259units-accs.csv");
    data_6 = xlsread("paper_results/smnist_results/smnist-6/smnist-6-final-258units-accs.csv");
    data_7 = xlsread("paper_results/smnist_results/smnist-7/smnist-7-final-259units-accs.csv");
    data_8 = xlsread("paper_results/smnist_results/smnist-8/smnist-8-final-258units-accs.csv");
    data_9 = xlsread("paper_results/smnist_results/smnist-9/smnist-9-final-259units-accs.csv");
    data_10 = xlsread("paper_results/smnist_results/smnist-10/smnist-10-final-123units-accs.csv");
    ylabel_text = "accuracy";
    offset = 0.01;
else
    results_file = fopen("stats_pattern-overalllosses.txt", 'w');
    
    data_1 = xlsread("paper_results/pattern_results/pattern-1/pattern-1-131units-accs.csv");
    data_2 = xlsread("paper_results/pattern_results/pattern-2/pattern-2-128units-accs.csv");
    data_3 = xlsread("paper_results/pattern_results/pattern-3/pattern-3-131units-accs.csv");
    data_4 = xlsread("paper_results/pattern_results/pattern-4/pattern-4-128units-accs.csv");
    data_5 = xlsread("paper_results/pattern_results/pattern-5/pattern-5-131units-accs.csv");
    data_6 = xlsread("paper_results/pattern_results/pattern-6/pattern-6-130units-accs.csv");
    data_7 = xlsread("paper_results/pattern_results/pattern-7/pattern-7-131units-accs.csv");
    data_8 = xlsread("paper_results/pattern_results/pattern-8/pattern-8-130units-accs.csv");
    data_9 = xlsread("paper_results/pattern_results/pattern-9/pattern-9-131units-accs.csv");
    data_10 = xlsread("paper_results/pattern_results/pattern-10/pattern-10-64units-accs.csv");
    
    ylabel_text = "MSE";
    offset = 0.1;
    if inset == 1
        offset = 0.025;
    end
end

data = [data_1.'; data_2.'; data_3.'; data_4.'; data_5.'; data_6.'; data_7.'; data_8.'; data_9.'; data_10.'].';
means_all = mean(data, 1)
stds_all = std(data, 1)

silence_props = linspace(0,1,6);

% RUN T-TESTS
[h,p] = ttest2(data_9, data_1);
fprintf(results_file, strcat('RNN vs het-nofurther: ', sprintf('%e', p)));
fprintf(results_file, '\n');

[h,p] = ttest2(data_9, data_2);
fprintf(results_file, strcat('RNN vs het-further: ', sprintf('%e', p)));
fprintf(results_file, '\n');

[h,p] = ttest2(data_9, data_3);
fprintf(results_file, strcat('RNN vs hom-further: ', sprintf('%e', p)));
fprintf(results_file, '\n');

[h,p] = ttest2(data_9, data_4);
fprintf(results_file, strcat('RNN vs hom-further: ', sprintf('%e', p)));
fprintf(results_file, '\n');

[h,p] = ttest2(data_9, data_5);
fprintf(results_file, strcat('RNN vs het-nofurther-noasc: ', sprintf('%e', p)));
fprintf(results_file, '\n');

[h,p] = ttest2(data_9, data_6);
fprintf(results_file, strcat('RNN vs het-further-noasc: ', sprintf('%e', p)));
fprintf(results_file, '\n');

[h,p] = ttest2(data_9, data_7);
fprintf(results_file, strcat('RNN vs hom-nofurther-noasc: ', sprintf('%e', p)));
fprintf(results_file, '\n');

[h,p] = ttest2(data_9, data_8);
fprintf(results_file, strcat('RNN vs hom-further-noasc: ', sprintf('%e', p)));
fprintf(results_file, '\n');

[h,p] = ttest2(data_9, data_10);
fprintf(results_file, strcat('RNN vs LSTM: ', sprintf('%e', p)));
fprintf(results_file, '\n');

[h,p] = ttest2(data_1, data_2);
fprintf(results_file, strcat('FHetA vs RHetA: ', sprintf('%e', p)));
fprintf(results_file, '\n');

[h,p] = ttest2(data_5, data_6);
fprintf(results_file, strcat('FHet vs RHet: ', sprintf('%e', p)));
fprintf(results_file, '\n');

[h,p] = ttest2(data_3, data_4);
fprintf(results_file, strcat('HomA vs LHetA: ', sprintf('%e', p)));
fprintf(results_file, '\n');

[h,p] = ttest2(data_7, data_8);
fprintf(results_file, strcat('Hom vs LHet: ', sprintf('%e', p)));
fprintf(results_file, '\n');

[h,p] = ttest2(data_1, data_5);
fprintf(results_file, strcat('FHet vs FHetA: ', sprintf('%e', p)));
fprintf(results_file, '\n');

[h,p] = ttest2(data_2, data_6);
fprintf(results_file, strcat('RHet vs RHetA: ', sprintf('%e', p)));
fprintf(results_file, '\n');

[h,p] = ttest2(data_3, data_7);
fprintf(results_file, strcat('Hom vs HomA: ', sprintf('%e', p)));
fprintf(results_file, '\n');

[h,p] = ttest2(data_4, data_8);
fprintf(results_file, strcat('LHet vs LHetA: ', sprintf('%e', p)));
fprintf(results_file, '\n');

[h,p] = ttest2(data_1, data_3);
fprintf(results_file, strcat('RHetA vs HomA: ', sprintf('%e', p)));
fprintf(results_file, '\n');

[h,p] = ttest2(data_5, data_7);
fprintf(results_file, strcat('RHet vs Hom: ', sprintf('%e', p)));
fprintf(results_file, '\n');

[h,p] = ttest2(data_2, data_4);
fprintf(results_file, strcat('RHetA vs LHetA: ', sprintf('%e', p)));
fprintf(results_file, '\n');

[h,p] = ttest2(data_6, data_8);
fprintf(results_file, strcat('RHet vs LHet: ', sprintf('%e', p)));
fprintf(results_file, '\n');

[h,p] = ttest2(data_5, data_7);
fprintf(results_file, strcat('FHet vs Hom: ', sprintf('%e', p)));
fprintf(results_file, '\n');

[h,p] = ttest2(data_6, data_8);
fprintf(results_file, strcat('RHet vs LHet: ', sprintf('%e', p)));
fprintf(results_file, '\n');

[h,p] = ttest2(data_1, data_4);
fprintf(results_file, strcat('FHetA vs LHetA: ', sprintf('%e', p)));
fprintf(results_file, '\n');

[h,p] = ttest2(data_5, data_8);
fprintf(results_file, strcat('FHet vs LHet: ', sprintf('%e', p)));
fprintf(results_file, '\n');

fprintf(results_file, '\n');

fclose(results_file);

% PLOT DATA
means = [means_all(9); means_all(10); means_all(3); means_all(7); means_all(1); means_all(5); means_all(2); means_all(6); means_all(4); means_all(8)]
stds = [stds_all(9); stds_all(10); stds_all(3); stds_all(7); stds_all(1); stds_all(5); stds_all(2); stds_all(6); stds_all(4); stds_all(8)];

model_names = {'RNN', 'LSTM', 'Hom', 'FHet', 'RHet', 'LHet'};
xticks_labels = [1 3 5.5 8.5, 11.5, 14.5];
xticks = [1 3 5 6 8 9 11 12 14 15];
asc_idx = [3, 5, 7, 9];
colors = ["#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677", "#AA4499", "#882255", "#72B803", "#109EC4", "#4DB8F6", "#4E1D87"];
colors_rbg = [51, 34, 136; 17, 119, 51; 68, 170, 153; 136, 204, 238; 221, 204, 119; 204, 102, 119; 
    170, 68, 153; 136, 34, 85; 114, 184, 3; 16, 158, 196; 77, 184, 246; 78, 29, 135] ./ 256;

if inset == 1
    means = means(3:end);
    stds = stds(3:end);
    model_names = {'Hom', 'FHet', 'RHet', 'LHet'};
    xticks_labels = [5.5 8.5, 11.5, 14.5];
    xticks = [5 6 8 9 11 12 14 15];
    asc_idx = [1, 3, 5, 7];
    colors = ["#44AA99", "#88CCEE", "#DDCC77", "#CC6677", "#AA4499", "#882255", "#72B803", "#109EC4", "#4DB8F6", "#4E1D87"];
    colors_rbg = [68, 170, 153; 136, 204, 238; 221, 204, 119; 204, 102, 119; 
        170, 68, 153; 136, 34, 85; 114, 184, 3; 16, 158, 196; 77, 184, 246; 78, 29, 135] ./ 256;

end
% means = [means(1) means(2) means(3) means(4) means(5) means(6)];
% stds = [stds(1) stds(2); stds(3) stds(4); stds(5) stds(6)];

b = bar(xticks, means, 1)

hold on

for i = asc_idx
    text(xticks(i), means(i) + stds(i) + offset, "A", 'FontName', 'helvetica', 'FontSize', fontsize-2, 'HorizontalAlignment', 'center');
end
hold on

ngroups = size(means, 1);
nbars = size(means, 2);
groupwidth = min(0.8, nbars/(nbars + 1.5));

for i = 1:nbars
    x = xticks - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
    errorbar(x, means.', stds.', 'k', 'linestyle', 'none', 'LineWidth', 2, 'HandleVisibility','off', 'CapSize', 10);
    hold on
end

b.FaceColor = 'flat';
for i = 1:ngroups
    b.CData(i, :) = colors_rbg(i, :);
end

if strcmp(task, "smnist")
    yticks(0.7:.1:1)
    ylim([0.8,1]);
end
if strcmp(task, "pattern")
    ylim([0,2.2])
    if inset == 1
        ylim([0,0.4]);
    end
end

set(gca,'XTick', xticks_labels, 'xticklabel', model_names, 'FontName', 'helvetica', 'FontSize', fontsize);

ylabel(ylabel_text, 'FontName', 'helvetica', 'FontSize', fontsize);

data_9_deldt = xlsread("paper_results/pattern_results/pattern-9/pattern-9-131units-accs.csv");
data_9_del1ms = xlsread("pattern_extra/pattern-9-final-delay1-131units-accs.csv");
mean(data_9_deldt)
mean(data_9_del1ms)
[h,p, ci, stats] = ttest2(data_9_deldt, data_9_del1ms, 'Tail', 'left') % test whether loss is lower when delay = dt is used
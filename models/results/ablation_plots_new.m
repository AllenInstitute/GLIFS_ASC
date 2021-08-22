clear all

fig = figure
fig.Renderer='Painters';

task = "pattern"; % smnist or pattern

results_file = fopen("stats_pattern.txt", 'w');
fontsize = 24;

if strcmp(task, "smnist")
    results_file = fopen("stats_smnist.txt", 'w');
    data_rnn = xlsread("results_wkof_080121/smnist-rnn-259units-0itr-ablation.csv");
    data_lstm = xlsread("results_wkof_080121/smnist-lstm-123units-0itr-ablation.csv");
    data_rglif = xlsread("results_wkof_080121/smnist-rglif-2asc-256units-0itr-ablation.csv");
    data_rglif_noasc = xlsread("results_wkof_080121/smnist-rglif-noasc-258units-0itr-ablation.csv");
    data_rglif_wtonly = xlsread("results_wkof_080121/smnist-rglif-wtonly-259units-0itr-ablation.csv");
    
    ylabel_text = "accuracy";
else
    results_file = fopen("stats_pattern.txt", 'w');
    
    data_1 = xlsread("results_wkof_080821/pattern-1-woreg-131units-0itr-ablation.csv");
    data_2 = xlsread("results_wkof_080821/pattern-2-woreg-128units-0itr-ablation.csv");
    data_3 = xlsread("results_wkof_080821/pattern-3-woreg-131units-0itr-ablation.csv");
    data_4 = xlsread("results_wkof_080821/pattern-4-woreg-128units-0itr-ablation.csv");
    data_5 = xlsread("results_wkof_080821/pattern-5-131units-0itr-ablation.csv");
    data_6 = xlsread("results_wkof_080821/pattern-6-130units-0itr-ablation.csv");
    data_7 = xlsread("results_wkof_080821/pattern-7-131units-0itr-ablation.csv");
    data_8 = xlsread("results_wkof_080821/pattern-8-130units-0itr-ablation.csv");
    data_9 = xlsread("results_wkof_080821/pattern-9-131units-0itr-ablation.csv");
    data_10 = xlsread("results_wkof_080821/pattern-10-64units-0itr-ablation.csv");
    
    ylabel_text = "MSE";
end

silence_props = linspace(0,1,6);

% RUN T-TESTS
for i = 1:size(data_1, 1)
    [h,p] = ttest2(data_9(i,:), data_1(i,:));
    fprintf(results_file, strcat('RNN vs het-nofurther: ', sprintf('%e', p)));
    fprintf(results_file, '\n');

    [h,p] = ttest2(data_9(i,:), data_2(i,:));
    fprintf(results_file, strcat('RNN vs het-further: ', sprintf('%e', p)));
    fprintf(results_file, '\n');

    [h,p] = ttest2(data_9(i,:), data_3(i,:));
    fprintf(results_file, strcat('RNN vs hom-further: ', sprintf('%e', p)));
    fprintf(results_file, '\n');

    [h,p] = ttest2(data_9(i,:), data_4(i,:));
    fprintf(results_file, strcat('RNN vs hom-further: ', sprintf('%e', p)));
    fprintf(results_file, '\n');

    [h,p] = ttest2(data_9(i,:), data_5(i,:));
    fprintf(results_file, strcat('RNN vs het-nofurther-noasc: ', sprintf('%e', p)));
    fprintf(results_file, '\n');

    [h,p] = ttest2(data_9(i,:), data_6(i,:));
    fprintf(results_file, strcat('RNN vs het-further-noasc: ', sprintf('%e', p)));
    fprintf(results_file, '\n');

    [h,p] = ttest2(data_9(i,:), data_7(i,:));
    fprintf(results_file, strcat('RNN vs hom-nofurther-noasc: ', sprintf('%e', p)));
    fprintf(results_file, '\n');

    [h,p] = ttest2(data_9(i,:), data_8(i,:));
    fprintf(results_file, strcat('RNN vs hom-further-noasc: ', sprintf('%e', p)));
    fprintf(results_file, '\n');

    [h,p] = ttest2(data_9(i,:), data_10(i,:));
    fprintf(results_file, strcat('RNN vs LSTM: ', sprintf('%e', p)));
    fprintf(results_file, '\n');

    [h,p] = ttest2(data_1(i,:), data_2(i,:));
    fprintf(results_file, strcat('het-nofurther vs het-further: ', sprintf('%e', p)));
    fprintf(results_file, '\n');

    [h,p] = ttest2(data_3(i,:), data_4(i,:));
    fprintf(results_file, strcat('hom-nofurther vs hom-further: ', sprintf('%e', p)));
    fprintf(results_file, '\n');

    [h,p] = ttest2(data_3(i,:), data_7(i,:));
    fprintf(results_file, strcat('hom-nofurther vs hom-nofurther-noasc: ', sprintf('%e', p)));
    fprintf(results_file, '\n');

    [h,p] = ttest2(data_4(i,:), data_8(i,:));
    fprintf(results_file, strcat('hom-further vs hom-further-noasc: ', sprintf('%e', p)));
    fprintf(results_file, '\n');

    [h,p] = ttest2(data_1(i,:), data_3(i,:));
    fprintf(results_file, strcat('het-nofurther vs hom-nofurther: ', sprintf('%e', p)));
    fprintf(results_file, '\n');

    [h,p] = ttest2(data_2(i,:), data_4(i,:));
    fprintf(results_file, strcat('het-further vs hom-further: ', sprintf('%e', p)));
    fprintf(results_file, '\n');

    [h,p] = ttest2(data_5(i,:), data_7(i,:));
    fprintf(results_file, strcat('het-nofurther-noasc vs hom-nofurther-noasc: ', sprintf('%e', p)));
    fprintf(results_file, '\n');

    [h,p] = ttest2(data_6(i,:), data_8(i,:));
    fprintf(results_file, strcat('het-further-noasc vs hom-further-noasc: ', sprintf('%e', p)));
    fprintf(results_file, '\n');

    [h,p] = ttest2(data_1(i,:), data_4(i,:));
    fprintf(results_file, strcat('het-nofurther vs hom-further: ', sprintf('%e', p)));
    fprintf(results_file, '\n');
    
    fprintf(results_file, '\n');
end
fclose(results_file);

% PLOT DATA
xticks = 1:6;

size(mean(data_9,2).')
means = [mean(data_9, 2).'; mean(data_10, 2).'; mean(data_3, 2).'; mean(data_7, 2).'; mean(data_1, 2).'; mean(data_5, 2).'; mean(data_4, 2).'; mean(data_8, 2).'].';
stds = [std(data_9, 0, 2).'; std(data_10, 0, 2).'; std(data_3, 0, 2).'; std(data_7, 0, 2).'; std(data_1, 0, 2).'; std(data_5, 0, 2).'; std(data_4, 0, 2).'; std(data_8, 0, 2).'].';
b = bar(xticks, means, 1)

hold on

ngroups = size(data_9, 1);
nbars = size(means, 2);
groupwidth = min(0.8, nbars/(nbars + 1.5));

for i = 1:nbars
    x = xticks - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
    errorbar(x, means(:,i), stds(:,i), 'k', 'LineWidth', 0.5, 'linestyle', 'none', 'HandleVisibility','off');
    hold on
end

colors = ["#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677", "#AA4499", "#882255"];

for i = 1:nbars
    b(i).FaceColor = colors(i);
end

set(gca,'XTick', 1:ngroups, 'xticklabel',silence_props, 'FontName', 'helvetica', 'FontSize', fontsize);
legend("RNN", "LSTM", "HOMA", "HOM", "HETIA", "HETI", "HETLA", "HETL", 'FontSize', fontsize);
xlabel("% silenced", 'FontSize', fontsize);
ylabel(ylabel_text);
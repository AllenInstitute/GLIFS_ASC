%% Produces Figure 6A
clear all

fig = figure
fig.Renderer='Painters';

fontsize = 24;
alpha = 0.05;

results_file = fopen("stats_smnist_ablation_dropout.txt", 'w');

data_1 = xlsread("paper_results/smnist_results/smnist-dropout/smnist-1-ablation-dropout-259units-ablated-accs.csv");
data_2 = xlsread("paper_results/smnist_results/smnist-dropout/smnist-2-ablation-dropout-256units-ablated-accs.csv");
data_3 = xlsread("paper_results/smnist_results/smnist-dropout/smnist-3-ablation-dropout-259units-ablated-accs.csv");
data_4 = xlsread("paper_results/smnist_results/smnist-dropout/smnist-4-ablation-dropout-256units-ablated-accs.csv");
data_5 = xlsread("paper_results/smnist_results/smnist-dropout/smnist-5-ablation-dropout-259units-ablated-accs.csv");
data_6 = xlsread("paper_results/smnist_results/smnist-dropout/smnist-6-ablation-dropout-258units-ablated-accs.csv");
data_7 = xlsread("paper_results/smnist_results/smnist-dropout/smnist-7-ablation-dropout-259units-ablated-accs.csv");
data_8 = xlsread("paper_results/smnist_results/smnist-dropout/smnist-8-ablation-dropout-258units-ablated-accs.csv");
data_9 = xlsread("paper_results/smnist_results/smnist-dropout/smnist-9-ablation-dropout-259units-ablated-accs.csv");
data_10 = xlsread("paper_results/smnist_results/smnist-dropout/smnist-10-ablation-dropout-123units-ablated-accs.csv");

ylabel_text = "accuracy";
offset = 0.01;

silence_props = linspace(0,1,6);

% RUN T-TESTS
p_values = zeros(size(data_1, 1), 10);
for i = 1:size(data_1, 1)
    [h,p] = ttest2(data_9(i,:), data_1(i,:));
    fprintf(results_file, strcat('RNN vs het-nofurther: ', sprintf('%e', p)));
    fprintf(results_file, '\n');
    p_values(i, 1) = p;

    [h,p] = ttest2(data_9(i,:), data_2(i,:));
    fprintf(results_file, strcat('RNN vs het-further: ', sprintf('%e', p)));
    fprintf(results_file, '\n');
    p_values(i, 2) = p;

    [h,p] = ttest2(data_9(i,:), data_3(i,:));
    fprintf(results_file, strcat('RNN vs hom-further: ', sprintf('%e', p)));
    fprintf(results_file, '\n');
    p_values(i, 3) = p;

    [h,p] = ttest2(data_9(i,:), data_4(i,:));
    fprintf(results_file, strcat('RNN vs hom-further: ', sprintf('%e', p)));
    fprintf(results_file, '\n');
    p_values(i, 4) = p;

    [h,p] = ttest2(data_9(i,:), data_5(i,:));
    fprintf(results_file, strcat('RNN vs het-nofurther-noasc: ', sprintf('%e', p)));
    fprintf(results_file, '\n');
    p_values(i, 5) = p;

    [h,p] = ttest2(data_9(i,:), data_6(i,:));
    fprintf(results_file, strcat('RNN vs het-further-noasc: ', sprintf('%e', p)));
    fprintf(results_file, '\n');
    p_values(i, 6) = p;

    [h,p] = ttest2(data_9(i,:), data_7(i,:));
    fprintf(results_file, strcat('RNN vs hom-nofurther-noasc: ', sprintf('%e', p)));
    fprintf(results_file, '\n');
    p_values(i, 7) = p;

    [h,p] = ttest2(data_9(i,:), data_8(i,:));
    fprintf(results_file, strcat('RNN vs hom-further-noasc: ', sprintf('%e', p)));
    fprintf(results_file, '\n');
    p_values(i, 8) = p;

    [h,p] = ttest2(data_9(i,:), data_10(i,:));
    fprintf(results_file, strcat('RNN vs LSTM: ', sprintf('%e', p)));
    fprintf(results_file, '\n');
    p_values(i, 10) = p;

    [h,p] = ttest2(data_1(i,:), data_2(i,:));
    fprintf(results_file, strcat('het-nofurther vs het-further: ', sprintf('%e', p)));
    fprintf(results_file, '\n');

    [h,p] = ttest2(data_1(i,:), data_5(i,:));
    fprintf(results_file, strcat('het-nofurther vs het-nofurther-noasc: ', sprintf('%e', p)));
    fprintf(results_file, '\n');
    
    [h,p] = ttest2(data_2(i,:), data_6(i,:));
    fprintf(results_file, strcat('het-further vs het-further-noasc: ', sprintf('%e', p)));
    fprintf(results_file, '\n');

    [h,p] = ttest2(data_4(i,:), data_8(i,:));
    fprintf(results_file, strcat('hom-further vs hom-further-noasc: ', sprintf('%e', p)));
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

p_values = [p_values(:, 9).'; p_values(:, 10).'; p_values(:, 3).'; p_values(:, 7).'; p_values(:, 1).'; p_values(:, 5).'; p_values(:, 2).'; p_values(:, 6).'; p_values(:, 4).'; p_values(:, 8).'; p_values(:, 9).'; p_values(:, 9).'].';
means = [mean(data_9, 2).'; mean(data_10, 2).'; mean(data_3, 2).'; mean(data_7, 2).'; mean(data_1, 2).'; mean(data_5, 2).'; mean(data_2, 2).'; mean(data_6, 2).'; mean(data_4, 2).'; mean(data_8, 2).'].';
stds = [std(data_9, 0, 2).'; std(data_10, 0, 2).'; std(data_3, 0, 2).'; std(data_7, 0, 2).'; std(data_1, 0, 2).'; std(data_5, 0, 2).'; std(data_2, 0, 2).'; std(data_6, 0, 2).'; std(data_4, 0, 2).'; std(data_8, 0, 2).'].';
b = bar(xticks, means, 1)

hold on

ngroups = size(data_9, 1);
nbars = size(means, 2);
groupwidth = min(0.8, nbars/(nbars + 1.5));

for i = 1:nbars
    x = xticks - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
    errorbar(x(2:5), means(2:5,i), stds(2:5,i), 'k', 'LineWidth', 1, 'CapSize', 10, 'linestyle', 'none', 'HandleVisibility','off');
    hold on
    
%     for j = 1:ngroups
%         if p_values(j, i) < alpha && p_values(j, i) ~= 0
%             text(x(j), means(j,i) + stds(j,i) + offset, "*", 'FontName', 'helvetica', 'FontSize', fontsize-2, 'HorizontalAlignment', 'center');
%         end
%     end
end

colors = ["#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677", "#AA4499", "#882255", "#72B803", "#109EC4", "#4DB8F6", "#4E1D87"];

for i = 1:nbars
    b(i).FaceColor = colors(i);
end

set(gca,'XTick', 1:ngroups, 'xticklabel',silence_props, 'FontName', 'helvetica', 'FontSize', fontsize);
legend("RNN", "LSTM", "HomA", "Hom", "FHetA", "FHet", "RHetA", "RHet", "LHetA", "LHet", 'FontSize', fontsize);
xlabel("ratio silenced", 'FontSize', fontsize);
ylabel(ylabel_text);
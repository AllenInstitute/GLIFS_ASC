fig = figure
fig.Renderer='Painters';

task = "smnist"; % smnist or pattern

results_file = fopen("stats_smnist-overalllosses.txt", 'w');

if strcmp(task, "smnist")
    data_rnn = xlsread("results_wkof_080121/smnist-rnn-259units-accs.csv");
    data_lstm = xlsread("results_wkof_080121/smnist-lstm-123units-accs.csv");
    data_rglif = xlsread("results_wkof_080121/smnist-rglif-2asc-256units-accs.csv");
    data_rglif_noasc = xlsread("results_wkof_080121/smnist-rglif-2asc-256units-accs.csv");
    data_rglif_wtonly = xlsread("results_wkof_080121/smnist-rglif-2asc-256units-accs.csv");
    
    ylabel_text = "accuracy";
else
    ylabel_text = "MSE";
end

silence_props = linspace(0,1,6);

% RUN T-TESTS
[h,p] = ttest2(data_rnn, data_lstm);
fprintf(results_file, strcat('LSTM: ', sprintf('%e', p)));
fprintf(results_file, '\n');

[h,p] = ttest2(data_rnn, data_rglif);
fprintf(results_file, strcat("RGLIF: ", sprintf('%e', p)));
fprintf(results_file, '\n');

[h,p] = ttest2(data_rnn, data_rglif_noasc);
fprintf(results_file, strcat("RGLIF_NoASC: ", sprintf('%e', p)));
fprintf(results_file, '\n');

[h,p] = ttest2(data_rnn, data_rglif_wtonly);
fprintf(results_file, strcat("RGLIF_WtOnly: ", sprintf('%e', p)));
fprintf(results_file, '\n');

fprintf(results_file, '\n');

fclose(results_file);

% PLOT DATA

mean_rnn = mean(data_rnn);
mean_lstm = mean(data_lstm);
mean_rglif = mean(data_rglif);
mean_rglif_noasc = mean(data_rglif_noasc);
mean_rglif_wtonly = mean(data_rglif_wtonly);

stds_rnn = std(data_rnn);
stds_lstm = std(data_lstm);
stds_rglif = std(data_rglif);
stds_rglif_noasc = std(data_rglif_noasc);
stds_rglif_wtonly = std(data_rglif_wtonly);

means = [mean_rnn; mean_lstm; mean_rglif; mean_rglif_noasc; mean_rglif_wtonly].';
stds = [stds_rnn; stds_lstm; stds_rglif; stds_rglif_noasc; stds_rglif_wtonly].';
b = bar(means)

hold on

ngroups = size(means, 2);
nbars = 1;
groupwidth = min(0.8, nbars/(nbars + 1.5));

for i = 1:nbars
    x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
    errorbar(x, means.', stds.', 'k', 'linestyle', 'none', 'HandleVisibility','off', 'CapSize', 10);
    hold on
end

colors = ["#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677", "#AA4499", "#882255"];
colors_rbg = [51, 34, 136;
    17, 119, 51;
    68, 170, 153;
    136, 204, 238;
    221, 204, 119;
    204, 102, 119;
    170, 68, 153;
    136, 34, 85] ./ 256;

b.FaceColor = 'flat';
for i = 1:ngroups
    b.CData(i, :) = colors_rbg(i, :);
end

set(gca,'XTick', 1:5, 'xticklabel',{'RNN', 'LSTM', 'RGLIF', 'RLIF', 'RGLIF_WT'}, 'FontName', 'helvetica', 'FontSize', 12);
ylabel(ylabel_text);
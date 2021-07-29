data_brnn = xlsread("results_final-brnn-initwithbursts-withdelay_256units_smnist_linebyline_lateralconns.csv");
data_brnn_noasc = xlsread("results_final-brnn-initwithbursts-withdelay-noasc_260units_smnist_linebyline_lateralconns.csv")
data_brnn_wtonly = xlsread("results_final-brnn-initwithbursts-withdelay-notrainparams_264units_smnist_linebyline_lateralconns.csv")
data_rnn = xlsread("results_final-rnn-initwithbursts-withoutdelay_264units_smnist_linebyline_lateralconns.csv");

means_brnn = mean(data_brnn, 2);
means_brnn_noasc = mean(data_brnn_noasc, 2);
means_brnn_wtonly = mean(data_brnn_wtonly, 2);
means_rnn = mean(data_rnn, 2);

stds_brnn = std(data_brnn, [], 2);
stds_brnn_noasc = std(data_brnn_noasc, [], 2);
stds_brnn_wtonly = std(data_brnn_wtonly, [], 2);
stds_rnn = std(data_rnn, [], 2);

means = [means_rnn.'; means_brnn.'; means_brnn_noasc.'; means_brnn_wtonly.'].';
stds = [stds_rnn.'; stds_brnn.'; stds_brnn_noasc.'; stds_brnn_wtonly.'].';
b = bar(means)

hold on

ngroups = 11;
nbars = 4;
groupwidth = min(0.8, nbars/(nbars + 1.5));

for i = 1:nbars
    x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
    errorbar(x, means(:,i), stds(:,i), 'k', 'linestyle', 'none', 'HandleVisibility','off');
    hold on
end

colors = ["#44AA99", "#88CCEE", "#DDCC77", "#CC6677"];

for i = 1:nbars
    b(i).FaceColor = colors(i);
end

set(gca,'XTick', 1:ngroups, 'xticklabel',linspace(0,1,11), 'FontName', 'helvetica', 'FontSize', 12);
legend("RNN", "RGLIF", "RLIF", "RGLIF_WT");
xlabel("% silenced");
ylabel("accuracy");
data_brnn = xlsread("results_mnist-ablate_brnn_smnist.csv");
data_rnn = xlsread("results_mnist-ablate_rnn_smnist.csv");

means_brnn = mean(data_brnn, 2);
means_rnn = mean(data_rnn, 2);
stds_brnn = std(data_brnn, [], 2);
stds_rnn = std(data_rnn, [], 2);

means = [means_brnn.'; means_rnn.'].';
stds = [stds_brnn.'; stds_rnn.'].';
b = bar(means)

hold on

ngroups = 11;
nbars = 2;
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
legend("BRNN", "RNN");
xlabel("% silenced");
ylabel("accuracy");
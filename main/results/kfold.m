clear all

% smnist_3 = xlsread("results_wkof_080821/smnist-3-sum-259units-kfold.csv");
% smnist_4 = xlsread("results_wkof_080821/smnist-4-sum-diffvals-090521-256units-kfold.csv");
% smnist_9 = xlsread("results_wkof_080821/smnist-9-sum-diffvalues-0905-259units-kfold.csv");
smnist_4 = xlsread("results_wkof_080821/smnist-4-dropout-256units-kfold-dropout.csv");

reg_lambda = [0, 0.2, 0.4, 0.6, 0.8, 1];

means = mean(smnist_4, 2);

stds = std(smnist_4, 0, 2);

means = [means.'].';
stds = [stds.'].';

b = bar(means)
% p = semilogx(reg_lambda, means, '.', 'MarkerSize', 20);
hold on

ngroups = size(means, 1);
nbars = size(means, 2);
groupwidth = min(0.8, nbars/(nbars + 1.5));
xticks = 1:ngroups;
xlabel("dropout probability")
ylabel("accuracy");

for i = 1:nbars
    size(means.')
    size(stds.')
    size(xticks)
    x = xticks - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
    errorbar(x, means(:,i).', stds(:,i).', 'k', 'linestyle', 'none', 'LineWidth', 1, 'HandleVisibility','off', 'CapSize', 10);
    hold on
end
set(gca,'XTick', 1:6, 'xticklabel',["0", "0.2", "0.4", "0.6", "0.8", "1.0"], 'FontName', 'helvetica', 'FontSize', 10);

figure
b = bar(means);
hold on

ngroups = size(means.', 1);
nbars = size(means.', 2);
groupwidth = min(0.8, nbars/(nbars + 1.5));
xticks = 1:ngroups;

for i = 1:nbars
%     x = xticks - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
    errorbar(x, means(i,:).', stds(i,:).', 'k', 'linestyle', 'none', 'LineWidth', 1, 'HandleVisibility','off', 'CapSize', 10);
    hold on
end
ylabel("accuracy");











% 
% 
% clear all
% 
% smnist_3 = xlsread("results_wkof_080821/smnist-3-sum-259units-kfold.csv");
% smnist_4 = xlsread("results_wkof_080821/smnist-4-sum-diffvals-090521-256units-kfold.csv");
% smnist_9 = xlsread("results_wkof_080821/smnist-9-sum-diffvalues-0905-259units-kfold.csv");
% smnist_10 = xlsread("results_wkof_080821/smnist-10-sum-123units-kfold.csv");
% 
% reg_lambda = [1e-20, 1e-15, 1e-10, 1e-5];
% 
% means_3 = mean(smnist_3, 2);
% means_4 = mean(smnist_4, 2);
% means_9 = mean(smnist_9, 2);
% means_10 = mean(smnist_10, 2);
% 
% stds_3 = std(smnist_3, 0, 2);
% stds_4 = std(smnist_4, 0, 2);
% stds_9 = std(smnist_9, 0, 2);
% stds_10 = std(smnist_10, 0, 2);
% 
% means = [means_4.'; means_9.'].';
% stds = [stds_4.'; stds_9.'].';
% 
% % b = bar(means)
% p = semilogx(reg_lambda, means, '.', 'MarkerSize', 20);
% xlim([1e-7, 1e-1]);
% hold on
% 
% ngroups = size(means, 1);
% nbars = size(means, 2);
% groupwidth = min(0.8, nbars/(nbars + 1.5));
% xticks = (reg_lambda);%1:ngroups;
% xlabel("reg lambda")
% ylabel("accuracy");
% 
% for i = 1:nbars
%     size(means.')
%     size(stds.')
%     size(xticks)
%     x = xticks;% - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
%     errorbar(x, means(:,i).', stds(:,i).', 'k', 'linestyle', 'none', 'LineWidth', 1, 'HandleVisibility','off', 'CapSize', 10);
%     hold on
% end
% legend(["RGLIF", "RNN"])
% 
% figure
% b = bar(means.');
% hold on
% 
% ngroups = size(means.', 1);
% nbars = size(means.', 2);
% groupwidth = min(0.8, nbars/(nbars + 1.5));
% xticks = 1:ngroups;
% 
% for i = 1:nbars
%     x = xticks - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
%     errorbar(x, means(i,:).', stds(i,:).', 'k', 'linestyle', 'none', 'LineWidth', 1, 'HandleVisibility','off', 'CapSize', 10);
%     hold on
% end
% set(gca,'XTick', 1:4, 'xticklabel',["RGLIF", "RNN"], 'FontName', 'helvetica', 'FontSize', 10);
% ylabel("accuracy");
% 
% legend(["1e-20", "1e-15", "1e-10", "1e-5"], 'FontName', 'helvetica', 'FontSize', 10, 'Location', 'northwest')
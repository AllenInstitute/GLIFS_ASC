losses = xlsread("results_wkof_080121/brnn_learnrealizable-membrane-1units-losses.csv");
final_outputs = xlsread("results_wkof_080121/brnn_learnrealizable-membrane-finaloutputs.csv");
targets = xlsread("results_wkof_080121/brnn_learnrealizable-membrane-targets.csv");
thresh_losses = xlsread("results_wkof_080121/brnn_learnrealizable-losses-threshes.csv");
km_losses = xlsread("results_wkof_080121/brnn_learnrealizable-losses-kms.csv");
asck_losses = xlsread("results_wkof_080121/brnn_learnrealizable-losses-asck.csv");
ascr_losses = xlsread("results_wkof_080121/brnn_learnrealizable-losses-ascr.csv");
ascamp_losses = xlsread("results_wkof_080121/brnn_learnrealizable-losses-ascamp.csv");

linewidth = 2;

figure;
plot(losses, 'Color', "#332288", 'LineWidth', linewidth);
xlabel('epoch #', 'FontName', 'helvetica', 'FontSize', 12);
ylabel('MSE', 'FontName', 'helvetica', 'FontSize', 12);

figure;
plot([0:0.05:3.95], final_outputs, 'Color', "#332288", 'LineWidth', linewidth, 'DisplayName', "learned");
xlabel('time (ms)', 'FontName', 'helvetica', 'FontSize', 12);
ylabel('firing probability', 'FontName', 'helvetica', 'FontSize', 12);
hold on
plot([0:0.05:3.95], targets, 'Color', "#117733", 'LineWidth', linewidth, 'DisplayName', "target");
xlabel('time (ms)', 'FontName', 'helvetica', 'FontSize', 12);
ylabel('firing probability', 'FontName', 'helvetica', 'FontSize', 12);

legend('FontName', 'helvetica', 'FontSize', 12, 'Location', 'northwest')

figure;
plot(thresh_losses(:,1), thresh_losses(:,2), 'Color', "#332288", 'LineWidth', linewidth);
xlabel('threshold (mV)', 'FontName', 'helvetica', 'FontSize', 12);
ylabel('MSE', 'FontName', 'helvetica', 'FontSize', 12);

figure;
plot(km_losses(:,1), km_losses(:,2), 'Color', "#332288", 'LineWidth', linewidth);
xlabel('membrane k (1/ms)', 'FontName', 'helvetica', 'FontSize', 12);
ylabel('MSE', 'FontName', 'helvetica', 'FontSize', 12);

figure;
plot(asck_losses(:,1), asck_losses(:,2), 'Color', "#332288", 'LineWidth', linewidth);
xlabel('ASC k (1/ms)', 'FontName', 'helvetica', 'FontSize', 12);
ylabel('MSE', 'FontName', 'helvetica', 'FontSize', 12);

figure;
plot(ascr_losses(:,1), ascr_losses(:,2), 'Color', "#332288", 'LineWidth', linewidth);
xlabel('ASC mult.', 'FontName', 'helvetica', 'FontSize', 12);
ylabel('MSE', 'FontName', 'helvetica', 'FontSize', 12);

figure;
plot(ascamp_losses(:,1), ascamp_losses(:,2), 'Color', "#332288", 'LineWidth', linewidth);
xlabel('ASC additive (pA)', 'FontName', 'helvetica', 'FontSize', 12);
ylabel('MSE', 'FontName', 'helvetica', 'FontSize', 12);
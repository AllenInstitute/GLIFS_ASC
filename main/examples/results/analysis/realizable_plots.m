%% Plot Figure 3
losses = xlsread("./../learn_realizable/brnn_learnrealizable-allparams-longer-agn-1units-losses.csv");
final_outputs = xlsread("./../learn_realizable/brnn_learnrealizable-allparams-longer-agn-finaloutputs.csv");
targets = xlsread("./../learn_realizable/brnn_learnrealizable-allparams-longer-agn-targets.csv");
initial_outputs = xlsread("./../learn_realizable/brnn_learnrealizable-allparams-longer-agn-initialoutputs.csv");

threshes = xlsread("./../learn_realizable/brnn_learnrealizable-allparams-longer-agn-1units-threshoverlearning.csv");
kms = xlsread("./../learn_realizable/brnn_learnrealizable-allparams-longer-agn-1units-kmoverlearning.csv");
ascks = xlsread("./../learn_realizable/brnn_learnrealizable-allparams-longer-agn-1units-asckoverlearning.csv");
ascrs = xlsread("./../learn_realizable/brnn_learnrealizable-allparams-longer-agn-1units-ascroverlearning.csv");
ascamps = xlsread("./../learn_realizable/brnn_learnrealizable-allparams-longer-agn-1units-ascampoverlearning.csv");

linewidth = 2;
simtime = 10;
fontsize=24;

% Loss of training
fig = figure
fig.Renderer='Painters';
subplot(1,2,1);
plot(losses, 'Color', "#332288", 'LineWidth', linewidth);
xlabel('epoch #', 'FontName', 'helvetica', 'FontSize', fontsize);
ylabel('MSE', 'FontName', 'helvetica', 'FontSize', fontsize);
set(gca,'FontSize', fontsize);

% Initial, final, target outputs
subplot(1,2,2);
plot([0:0.05:simtime-0.05], initial_outputs, 'Color', "#332288", 'LineWidth', linewidth, 'DisplayName', "initial");
xlabel('time (ms)', 'FontName', 'helvetica', 'FontSize', fontsize);
ylabel('firing', 'FontName', 'helvetica', 'FontSize', fontsize);
hold on
plot([0:0.05:simtime-0.05], final_outputs, 'Color', "#117733", 'LineWidth', linewidth, 'DisplayName', "learned");
xlabel('time (ms)', 'FontName', 'helvetica', 'FontSize', fontsize);
ylabel('firing', 'FontName', 'helvetica', 'FontSize', fontsize);
hold on
plot([0:0.05:simtime - 0.05], targets, 'Color', "#88CCEE", 'LineWidth', linewidth, 'DisplayName', "target");
xlabel('time (ms)', 'FontName', 'helvetica', 'FontSize', fontsize);
ylabel('firing rate', 'FontName', 'helvetica', 'FontSize', fontsize);
set(gca,'FontSize', fontsize);
legend('FontName', 'helvetica', 'FontSize', fontsize, 'Location', 'best')

% Parameters in network over training
fig = figure
fig.Renderer='Painters';
colors = ["#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677", "#AA4499", "#882255"];
plot(threshes, 'Color', colors(1), 'LineWidth', linewidth, 'DisplayName', "thresh (mV)")
hold on
plot(kms, 'Color', colors(2), 'LineWidth', linewidth, 'DisplayName', "k_m (1/ms)")
hold on
plot(ascks(:,1), 'Color', colors(3), 'LineWidth', linewidth, 'DisplayName', "k_j (1/ms)")
plot(ascks(:,2), 'Color', colors(3), 'LineWidth', linewidth, 'HandleVisibility', "off")
hold on
plot(ascamps(:,1), 'Color', colors(4), 'LineWidth', linewidth, 'DisplayName', "a_j (mV)")
plot(ascamps(:,2), 'Color', colors(4), 'LineWidth', linewidth, 'HandleVisibility', "off")
hold on
plot(ascrs(:,1), 'Color', colors(5), 'LineWidth', linewidth, 'DisplayName', "r_j (mV)")
plot(ascrs(:,2), 'Color', colors(5), 'LineWidth', linewidth, 'HandleVisibility', "off")
hold on
yline(0, 'LineWidth', linewidth, 'HandleVisibility', "off")
xlabel('epoch #', 'FontName', 'helvetica', 'FontSize', fontsize);
ylabel('difference from target', 'FontName', 'helvetica', 'FontSize', fontsize);
set(gca,'FontSize', fontsize);
legend('FontName', 'helvetica', 'FontSize', fontsize, 'Location', 'best')
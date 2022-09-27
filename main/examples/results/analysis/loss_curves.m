%% Produces Figure 6 and Figure 8
% Figure stored to {task}_losscurve.svg

clearvars
fig = figure('Position', get(0, 'Screensize'));
fig.Renderer='Painters';

% TODO: Change below line
task = "lmnist-lowsigma"; % sine, lmnist, nmnist, pmnist, lmnist-lowsigmav, lmnist-anneal
linewidth = 2;
fontsize=24;

ylabeltext = "MSE";
if strcmp(task, "pmnist") || strcmp(task, "nmnist") || strcmp(task, "lmnist") || strcmp(task, "lmnist-anneal") || strcmp(task, "lmnist-lowsigma")
    ylabeltext = "cross entropy loss";
end
colors_rbg = [51, 34, 136;
    17, 119, 51;
    68, 170, 153;
    136, 204, 238;
    221, 204, 119;
    204, 102, 119;
    170, 68, 153;
    136, 34, 85;
    114, 184, 3;
    16, 158, 196;
    77, 184, 246;
    78, 29, 135] ./ 256;
colors = ["#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677", "#AA4499", "#882255", "#72B803", "#109EC4", "#4DB8F6", "#4E1D87"];
network_types = ["rnn", "glifr_homa", "glifr_lheta", "glifr_fheta", "glifr_rheta"];
network_names = ["RNN", "HomA", "LHetA", "FHetA", "RHetA"];

if strcmp(task, "pmnist")
    network_types = ["rnn", "glifr_lheta", "glifr_rheta"];
    network_names = ["RNN", "LHetA", "RHetA"];
end
if strcmp(task, "lmnist-anneal") || strcmp(task, "lmnist-lowsigma")
    network_types = ["glifr_lheta", "glifr_rheta"];
    network_names = ["LHetA", "RHetA"];
end
ps = [];
for r = 1:length(network_types)
    n = network_types(r);
    losses = [];
    d = dir("./../" + task + "/" + n + "/*/*/*/train_loss.csv")
    for didx = 1:size(d,1)
        filename = d(didx).folder + "\" + d(didx).name;
        losses = [losses; readmatrix(filename)];
    end
    losses = losses.';
    num_epochs = size(losses, 1);
    
    stdev = reshape(smoothdata(std(losses, 0, 2)), 1, num_epochs);
    avg = reshape(mean(losses, 2), 1, num_epochs);
    
    x = [1:numel(avg)];
    top = avg + stdev;
    bot = avg - stdev;
    xrev = fliplr(x);
    botrev = fliplr(bot);
    patch([x, xrev], [top, botrev], colors_rbg(r), 'FaceColor', colors(r), 'EdgeColor', colors(r), 'FaceAlpha', 0.5, 'HandleVisibility', 'off')%colors_rbg(r));
    hold on;
    ps(end + 1) = plot(x, avg, 'Color', colors(r), 'LineWidth', linewidth);
    hold on
    if strcmp(task, "lmnist-anneal") || strcmp(task, "lmnist-lowsigma")
        ylim([0, 2.5]);
    end
end
    if strcmp(task, "lmnist-anneal") || strcmp(task, "lmnist-lowsigma")
        ylim([0, 2.5]);
    end

set(gca,'FontName', 'helvetica', 'FontSize', fontsize);
xlim([0,num_epochs]);

legend(network_names, 'FontSize', fontsize, 'Location', 'northeast');
xlabel("epoch #", 'FontName', 'helvetica', 'FontSize', fontsize);
ylabel(ylabeltext, 'FontName', 'helvetica', 'FontSize', fontsize);

saveas(fig, task + "_losscurve.svg",'svg');
close(fig);
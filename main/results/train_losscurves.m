%% Produces Figure 4E, 5C

clearvars
fig = figure
fig.Renderer='Painters';

task = "smnist"; %pattern or smnist
linewidth = 2;
fontsize=24;

ylabeltext = "MSE";
if strcmp(task, "smnist")
    ylabeltext = "cross entropy loss";
end
params = [131,128,131,128,131,130,131,130,131,64];
if strcmp(task, "smnist")
    params = [259, 256, 259, 256, 259, 258, 259, 258, 259, 123];
end
num_epochs = 5000;
if strcmp(task, "smnist")
    num_epochs = 50;
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
order = uint8([9, 10, 3, 7, 1, 5, 2, 6, 4, 8]);

ps = [];
for r = 1:10
    i = order(r);
    losses = zeros(num_epochs, 30);
    for j = 0:29
        if strcmp(task, "pattern")
            filename = "paper_results/pattern_results/pattern-" + string(i) + "/pattern-" + string(i) + "-" + params(i) + "units-" + string(j) + "itr-losses.csv";
            if j > 19
                filename = "paper_results/pattern_results/pattern-" + string(i) + "/pattern-" + string(i) + "-moretrials-" + params(i) + "units-" + string(j - 20) + "itr-losses.csv";
            end
        elseif strcmp(task, "smnist")
            filename = "paper_results/smnist_results/smnist-" + string(i) + "/smnist-" + string(i) + "-final-" + params(i) + "units-" + string(j) + "itr-losses.csv";
        end
        losses(:,j+1) = xlsread(filename);
    end
    
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
end
set(gca,'FontName', 'helvetica', 'FontSize', fontsize);
xlim([0,num_epochs]);
legend("RNN", "LSTM", "HomA", "Hom", "FHetA", "FHet", "RHetA", "RHet", "LHetA", "LHet", 'FontSize', fontsize, 'Location', 'eastoutside');
xlabel("epoch #", 'FontName', 'helvetica', 'FontSize', fontsize);
ylabel(ylabeltext, 'FontName', 'helvetica', 'FontSize', fontsize);
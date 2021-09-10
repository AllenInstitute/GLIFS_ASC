fig = figure
fig.Renderer='Painters';

task = "smnist"; % smnist or pattern
fontsize = 24;
linewidth = 2;

name = "4-agn";
clustername = "smnist-4-agn-256units-256units-0itr-allparams-clusters";
paramname = "smnist-4-agn-256units-256units-0itr-allparams";
initparamname = "smnist-4-agn-256units-init-256units-0itr-allparams";

clusters = xlsread(strcat("results_wkof_080821/", clustername, ".csv"));
parameters = xlsread(strcat("results_wkof_080821/", paramname, ".csv"));
initparameters = xlsread(strcat("results_wkof_080821/", initparamname, ".csv"));
    
num_neurons = size(parameters, 1);

xlim([-5000, 5000]);
xlabel('current (pA)', 'FontName', 'helvetica', 'FontSize', fontsize);
ylabel('avg. firing probability', 'FontName', 'helvetica', 'FontSize', fontsize);
set(gca,'FontSize', fontsize);

fig = figure
fig.Renderer='Painters';

subplot(2,2,1)
scatter(initparameters(:,1), initparameters(:,2), [], clusters);
xlabel('threshold (mV)', 'FontName', 'helvetica', 'FontSize', fontsize);
ylabel('km (ms^-1)', 'FontName', 'helvetica', 'FontSize', fontsize);
set(gca,'FontSize', fontsize);

subplot(2,2,2)
fig.Renderer='Painters';
scatter(initparameters(:,3), initparameters(:,4), [], clusters);
xlabel('r1', 'FontName', 'helvetica', 'FontSize', fontsize);
ylabel('r2', 'FontName', 'helvetica', 'FontSize', fontsize);
set(gca,'FontSize', fontsize);

subplot(2,2,3)
fig.Renderer='Painters';
scatter(initparameters(:,5), initparameters(:,6), [], clusters);
xlabel('k1', 'FontName', 'helvetica', 'FontSize', fontsize);
ylabel('k2', 'FontName', 'helvetica', 'FontSize', fontsize);
set(gca,'FontSize', fontsize);

subplot(2,2,4)
fig.Renderer='Painters';
scatter(initparameters(:,7), initparameters(:,8), [], clusters);
xlabel('a1', 'FontName', 'helvetica', 'FontSize', fontsize);
ylabel('a2', 'FontName', 'helvetica', 'FontSize', fontsize);
set(gca,'FontSize', fontsize);

fig = figure
fig.Renderer='Painters';

subplot(2,2,1)
scatter(parameters(:,1), parameters(:,2), [], clusters);
xlabel('threshold (mV)', 'FontName', 'helvetica', 'FontSize', fontsize);
ylabel('km (ms^-1)', 'FontName', 'helvetica', 'FontSize', fontsize);
set(gca,'FontSize', fontsize);

subplot(2,2,2)
fig.Renderer='Painters';
scatter(parameters(:,3), parameters(:,4), [], clusters);
xlabel('r1', 'FontName', 'helvetica', 'FontSize', fontsize);
ylabel('r2', 'FontName', 'helvetica', 'FontSize', fontsize);
set(gca,'FontSize', fontsize);

subplot(2,2,3)
fig.Renderer='Painters';
scatter(parameters(:,5), parameters(:,6), [], clusters);
xlabel('k1', 'FontName', 'helvetica', 'FontSize', fontsize);
ylabel('k2', 'FontName', 'helvetica', 'FontSize', fontsize);
set(gca,'FontSize', fontsize);

subplot(2,2,4)
fig.Renderer='Painters';
scatter(parameters(:,7), parameters(:,8), [], clusters);
xlabel('a1', 'FontName', 'helvetica', 'FontSize', fontsize);
ylabel('a2', 'FontName', 'helvetica', 'FontSize', fontsize);
set(gca,'FontSize', fontsize);
% nbins = 20;
% 
% histogram(slopes, nbins);
% xlabel("f/I slope", 'FontSize', fontsize);

colors = ['r', 'g', 'b', 'm'];
ylabels = ["thresh", "km", "r", "k", "a"];
fig = figure
fig.Renderer='Painters';
for i = [1:8]
    subplot(3,3,i)
    fig.Renderer='Painters';
    for j = [1:num_neurons]
        quiver(1, initparameters(j, i), 1, parameters(j,i) - initparameters(j,i), colors(clusters(j) + 1));
        hold on
    end
    ylabel(ylabels(i), 'FontName', 'helvetica', 'FontSize', fontsize);
    set(gca,'FontSize', fontsize);
end
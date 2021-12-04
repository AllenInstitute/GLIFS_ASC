clear all
% % fig = figure
% % fig.Renderer='Painters';

task = "smnist"; % smnist or pattern
fontsize = 24;
linewidth = 2;

ch_scores = xlsread("results_wkof_080821/smnist-4-final-256units-0itr-allparams-chscores.csv")
scatter(ch_scores(:, 1), ch_scores(:, 2), 'k', 'filled');
xlabel('# clusters', 'FontName', 'helvetica', 'FontSize', fontsize);
ylabel('CH index', 'FontName', 'helvetica', 'FontSize', fontsize);
set(gca,'FontSize', fontsize);
ylim([150, 220]);

name = "4-agn";
% clustername = "smnist-4-agn-256units-256units-0itr-allparams-clusters";
% paramname = "smnist-4-agn-256units-256units-0itr-allparams";
% initparamname = "smnist-4-agn-256units-init-256units-0itr-allparams";
% outputsname = "smnist-4-agn-sampleoutputs";

clustername = "smnist-4-final-256units-0itr-allparams-clusters";
paramname = "smnist-4-final-256units-0itr-allparams";

% initparamname = "smnist-2-agn-256units-init-256units-0itr-allparams";
% outputsname = "smnist-4-agn-sampleoutputs";

clusters_labels = xlsread(strcat("results_wkof_080821/", clustername, ".csv"));
parameters = xlsread(strcat("results_wkof_080821/", paramname, ".csv"));
% outputs = xlsread(strcat("results_wkof_080821/", outputsname, ".csv"));
% initparameters = xlsread(strcat("results_wkof_080821/", initparamname, ".csv"));
  
num_neurons = size(parameters, 1);

% fig = figure
% fig.Renderer='Painters';
% 
% subplot(2,2,1)
% scatter(initparameters(:,1), initparameters(:,2), [], clusters);
% xlabel('threshold (mV)', 'FontName', 'helvetica', 'FontSize', fontsize);
% ylabel('km (ms^-1)', 'FontName', 'helvetica', 'FontSize', fontsize);
% set(gca,'FontSize', fontsize);
% 
% subplot(2,2,2)
% fig.Renderer='Painters';
% scatter(initparameters(:,3), initparameters(:,4), [], clusters);
% xlabel('r1', 'FontName', 'helvetica', 'FontSize', fontsize);
% ylabel('r2', 'FontName', 'helvetica', 'FontSize', fontsize);
% set(gca,'FontSize', fontsize);
% 
% subplot(2,2,3)
% fig.Renderer='Painters';
% scatter(initparameters(:,5), initparameters(:,6), [], clusters);
% xlabel('k1', 'FontName', 'helvetica', 'FontSize', fontsize);
% ylabel('k2', 'FontName', 'helvetica', 'FontSize', fontsize);
% set(gca,'FontSize', fontsize);
% 
% subplot(2,2,4)
% fig.Renderer='Painters';
% scatter(initparameters(:,7), initparameters(:,8), [], clusters);
% xlabel('a1', 'FontName', 'helvetica', 'FontSize', fontsize);
% ylabel('a2', 'FontName', 'helvetica', 'FontSize', fontsize);
% set(gca,'FontSize', fontsize);

fig = figure
fig.Renderer='Painters';

colors_rbg = [51, 34, 136;
    68, 170, 153;
    221, 204, 119;
    204, 102, 119] ./ 256;

subplot(2,2,1)
scatter(parameters(:,1), parameters(:,2), [], clusters_labels, 'filled', 'MarkerFaceAlpha',.2);
xlabel('thresh (mV)', 'FontName', 'helvetica', 'FontSize', fontsize);
ylabel('k_m (1/ms)', 'FontName', 'helvetica', 'FontSize', fontsize);
set(gca,'FontSize', fontsize);
colormap(colors_rbg)

subplot(2,2,2)
fig.Renderer='Painters';
scatter(parameters(:,3), parameters(:,4), [], clusters_labels, 'filled', 'MarkerFaceAlpha',.2);
% hold on
% h=fill([-0.01,0.01,0.01,-0.01],[-0.01,-0.01,0.01,0.01],'red');
% h.FaceAlpha=0.3;
xlabel('r_1', 'FontName', 'helvetica', 'FontSize', fontsize);
ylabel('r_2', 'FontName', 'helvetica', 'FontSize', fontsize);
set(gca,'FontSize', fontsize);

subplot(2,2,3)
fig.Renderer='Painters';
scatter(parameters(:,5), parameters(:,6), [], clusters_labels, 'filled', 'MarkerFaceAlpha',.2);
xlabel('k_1 (1/ms)', 'FontName', 'helvetica', 'FontSize', fontsize);
ylabel('k_2 (1/ms)', 'FontName', 'helvetica', 'FontSize', fontsize);
set(gca,'FontSize', fontsize);

subplot(2,2,4)
fig.Renderer='Painters';
scatter(parameters(:,7), parameters(:,8), [], clusters_labels, 'filled', 'MarkerFaceAlpha',.2);
% hold on
% h=fill([-0.01,0.01,0.01,-0.01],[-0.01,-0.01,0.01,0.01],'red');
% h.FaceAlpha=0.3;
xlabel('a_1 (pA)', 'FontName', 'helvetica', 'FontSize', fontsize);
ylabel('a_2 (pA)', 'FontName', 'helvetica', 'FontSize', fontsize);
set(gca,'FontSize', fontsize);
% nbins = 20;
% 
% histogram(slopes, nbins);
% xlabel("f/I slope", 'FontSize', fontsize);

% colors = ['r', 'g', 'b', 'm'];
% ylabels = ["thresh", "km", "r1", "r2", "k1", "k2", "a1", "a2"];
% fig = figure
% fig.Renderer='Painters';
% for i = [1:8]
%     subplot(3,3,i)
%     fig.Renderer='Painters';
%     for j = [1:num_neurons]
%         quiver(1, initparameters(j, i), 1, parameters(j,i) - initparameters(j,i), colors(clusters(j) + 1));
%         hold on
%     end
%     ylabel(ylabels(i), 'FontName', 'helvetica', 'FontSize', fontsize);
%     set(gca,'FontSize', fontsize);
% end

% fig = figure
% fig.Renderer='Painters';
% sim_time = 5;
% fig.Renderer='Painters';
% for j = [1:10]%num_neurons]
%     subplot(1,3,clusters(j) + 1);
%     plot([0:0.05:sim_time-0.05], outputs(:, j), 'LineWidth', linewidth);%, 'Color', colors(clusters(j) + 1));
%     hold on
% end
% ylabel("firing", 'FontName', 'helvetica', 'FontSize', fontsize);
% set(gca,'FontSize', fontsize);
solve_time = load('solve_time.mat');
gmpc_solver_time = solve_time.gmpc_solver_time;
nmpc_solver_time = solve_time.nmpc_solver_time;

% calculate mean and std
gmpc_solver_time_mean = mean(gmpc_solver_time);
gmpc_solver_time_std = std(gmpc_solver_time - gmpc_solver_time_mean);
nmpc_solver_time_mean = mean(nmpc_solver_time);
nmpc_solver_time_std = std(nmpc_solver_time - nmpc_solver_time_mean);

% max and min
gmpc_solver_time_max = max(gmpc_solver_time);
gmpc_solver_time_min = min(gmpc_solver_time);
nmpc_solver_time_max = max(nmpc_solver_time);
nmpc_solver_time_min = min(nmpc_solver_time);

fprintf('gmpc_solver_time_mean: %f\n', gmpc_solver_time_mean);
fprintf('gmpc_solver_time_std: %f\n', gmpc_solver_time_std);
fprintf('nmpc_solver_time_mean: %f\n', nmpc_solver_time_mean);
fprintf('nmpc_solver_time_std: %f\n', nmpc_solver_time_std);
fprintf('gmpc_solver_time_max: %f\n', gmpc_solver_time_max);
fprintf('gmpc_solver_time_min: %f\n', gmpc_solver_time_min);
fprintf('nmpc_solver_time_max: %f\n', nmpc_solver_time_max);
fprintf('nmpc_solver_time_min: %f\n', nmpc_solver_time_min);

h= figure;

labels = {'GMPC', 'NMPC'};

font_size = 18;
line_width = 2;
set(h, 'DefaultTextFontSize', font_size);
boxplot([gmpc_solver_time', nmpc_solver_time'], 'Labels', {'GMPC', 'NMPC'}, 'Whisker', 1, 'Colors', 'b', 'symbol','');
% Set the fill color for each box
h_box=findobj(gca,'Tag','Box');
h_M = findobj(gca,'Tag','Median');

color(1,:) = [255 5 5]/255;
color(2,:) = [255 100 5]/255;
color(3,:) = [255 255 5]/255;
color(4,:) = [200 255 5]/255;
color(5,:) = [5 255 5]/255;
color(6,:) = [5 255 100]/255;
color(7,:) = [5 255 255]/255;
color(8,:) = [5 200 255]/255;
color(9,:) = [5 5 255]/255;
color(10,:) = [255 5 255]/255;
for j=1:length(h_box)
    h_M(j).Color='k';
    uistack(patch(get(h_box(j),'XData'),get(h_box(j),'YData'),color(6),'FaceAlpha',1,'linewidth',1.5),'bottom');
end
ylim([0.012  0.035])
ylabel('Solve time (s)');
set(gca,'XTick',1:2)%2 because only exists 2 boxplot
set(gca,'XTickLabel',{'GMPC', 'NMPC'},'FontSize',14)
% tightfig;
% saveas(gcf, 'solve_time.jpg');
% close(gcf);
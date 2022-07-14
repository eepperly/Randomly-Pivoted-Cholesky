method_names = { 'RLS', 'Uniform', 'Greedy', 'RPCholesky'};
method_names_for_plot = { 'RLS', 'Uniform', 'Greedy', 'RPCholesky'};
ks = 200:200:1000;
specs = {'--', ':', '*-', '-.'};
colors = {[0.8500 0.3250 0.0980], [0.9290 0.6940 0.1250], ...
    [0.4940 0.1840 0.5560], [0.4660 0.6740 0.1880] };

figure(1)
for j = 1:length(method_names)
    method_name = method_names{j};
    load(sprintf('../data/%s_molecule100k.mat', method_name));

    if strcmp(method_name, 'Greedy')
        figure(1)
        plot(ks, KRRSMAPE(:,1), specs{j},...
            'LineWidth', 2', 'Color', colors{j})
        hold on
        figure(2)
        plot(ks, KRRMAE(:,1), specs{j},...
            'LineWidth', 2', 'Color', colors{j})
        hold on
        figure(3)
        plot(ks, trace_errors(:,1), specs{j},...
            'LineWidth', 2', 'Color', colors{j})
        hold on
    else
        figure(1)
        errorbar(ks, KRRSMAPE(:,1), KRRSMAPE(:,2), specs{j},...
            'LineWidth', 2', 'Color', colors{j})
        hold on
        figure(2)
        errorbar(ks, KRRMAE(:,1), KRRMAE(:,2), specs{j},...
            'LineWidth', 2', 'Color', colors{j})
        hold on
        figure(3)
        errorbar(ks, trace_errors(:,1), trace_errors(:,2), specs{j},...
            'LineWidth', 2', 'Color', colors{j})
        hold on
    end
end

figure(1)
legend(method_names_for_plot, 'location', 'best', 'FontSize', 26)
xlabel('$k$','FontSize', 26)
ylabel('SMAPE', 'FontSize', 26)
ax = gca;
ax.FontSize = 20; 
saveas(gcf, '../figs/chemistry_smape.png');

figure(2)
xlabel('$k$','FontSize', 26)
ylabel('MAE (eV)', 'FontSize', 26)
ax = gca;
ax.FontSize = 20; 
saveas(gcf, '../figs/chemistry_mae.png');

figure(3)
xlabel('$k$','FontSize', 26)
ylabel('Relative Trace-Norm Error', 'FontSize', 26)
ax = gca;
ax.FontSize = 20; 
saveas(gcf, '../figs/chemistry_errors.png');
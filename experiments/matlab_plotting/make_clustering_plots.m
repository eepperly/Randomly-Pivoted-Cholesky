method_names = { 'DPP', 'RLS', 'Uniform', 'Greedy', 'RPCholesky' };
method_names_for_plot = { 'DPP', 'RLS', 'Uniform', 'Greedy', 'RPCholesky'};
ks = 0:20:200;
specs = {'-', '--', ':', '*-', '-.'};
colors = {[0 0.4470 0.7410], [0.8500 0.3250 0.0980],...
    [0.9290 0.6940 0.1250], ...
    [0.4940 0.1840 0.5560], [0.4660 0.6740 0.1880]};

figure(1)
for j = 1:length(method_names)
    method_name = method_names{j};
    load(sprintf('../data/%s_accuracies.mat', method_name));
    accuracies(isnan(accuracies)) = 0;
    accuracies = sum(accuracies == 1,2) / size(accuracies, 2);
    plot(ks, [0;accuracies], specs{j}, 'LineWidth', 2, 'Color', colors{j})
    
    hold on
end

legend(method_names_for_plot, 'location', 'best', 'FontSize', 26)
ax = gca;
ax.FontSize = 20; 
xlabel('$k$','FontSize', 26)
ylabel('Perfect Recovery Fraction', 'FontSize', 26)
axis([0 200 0.0 1.1])
saveas(gcf, '../figs/accuracies.png');

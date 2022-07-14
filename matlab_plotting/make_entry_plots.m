method_names = { 'DPP', 'RLS', 'Uniform', 'Greedy', 'RPCholesky'};
method_names_for_plot = { 'DPP', 'RLS', 'Uniform', 'Greedy', 'RPCholesky'};
ks = 0:20:100;
specs = {'-', '--', ':', '*-', '-.'};
names = {'smile', 'outliers'};

for l = 1:2
    figure(l)
    for j = 1:length(method_names)
        method_name = method_names{j};
        load(sprintf('../data/%s_%s_queries.mat', names{l}, method_name));
        queries = queries / 10000;

        if strcmp(method_name, 'DPP') || strcmp(method_name, 'RLS')
            errorbar(ks, mean(queries, 2) ./ ks', std(queries, 1, 2) ./ ks',...
                specs{j}, 'LineWidth', 2')
        else
            plot(ks, mean(queries, 2) ./ ks', specs{j}, 'LineWidth', 2)
        end
        hold on
    end
    
    legend(method_names_for_plot, 'location', 'best', 'FontSize', 26)
    xlabel('$k$','FontSize', 26)
    ylabel('(Entry evaluations) / $Nk$', 'FontSize', 26)
    set(gca, 'YScale', 'log')
    axis([20 100 0.5 1000])
    ax = gca; ax.FontSize = 20; 
    saveas(gcf, sprintf('../figs/entries_%s.png', names{l}));
end

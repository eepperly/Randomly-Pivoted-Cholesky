method_names = { 'DPP', 'RLS', 'Uniform', 'Greedy', 'RPCholesky'};
method_names_for_plot = { 'DPP', 'RLS', 'Uniform', 'Greedy', 'RPCholesky'};
ks = 0:10:150;
specs = {'-', '--', ':', '*-', '-.'};
names = {'smile', 'spiral'};
colors = {"#0072BD","#D95319","#EDB120","#7E2F8E","#77AC30"};
order = [1 2 4 5 3];

close all
for l = 1:2
    figure(l)
    for j = 1:length(method_names)
        idx = order(j);
        method_name = method_names{idx};
        load(sprintf('../data/%s_%s_queries.mat', names{l}, method_name));
        queries = queries / 10000;

        if strcmp(method_name, 'DPP') || strcmp(method_name, 'RLS')
            errorbar(ks, mean(queries, 2) ./ ks', std(queries, 1, 2) ./ ks',...
                specs{j}, 'LineWidth', 2', 'Color', colors{idx})
        else
            plot(ks, mean(queries, 2) ./ ks', specs{idx}, 'LineWidth',...
                2, 'Color', colors{idx}, 'MarkerFaceColor', colors{idx})
        end
        hold on
    end
    
    if l == 1
        legend(method_names_for_plot{order}, 'location', 'northeast',...
            'FontSize', 26)
    end
    xlabel('Rank $k$','FontSize', 26)
    ylabel('(Entry evaluations) / $Nk$', 'FontSize', 26)
    set(gca, 'YScale', 'log')
    axis([10 150 0.5 1000])
    ax = gca; ax.FontSize = 20; 
    saveas(gcf, sprintf('../figs/entries_%s.png', names{l}));
end

matrix_names = {'smile', 'spiral'};
method_names = {'DPP', 'RLS', 'Uniform', 'Greedy', 'RPCholesky'};
method_names_for_plot = {'DPP', 'RLS', 'Uniform', 'Greedy', 'RPCholesky'};
colors = {"#0072BD","#D95319","#EDB120","#7E2F8E","#77AC30"};
specs = {'-', '--', ':', '*-', '-.'};
order = [3 4 5];
% order = [3 2 1 4 5];
last_idx = 15;
ks = 10:10:(10*last_idx);

close all
for i = 1:length(matrix_names)
    matrix_name = matrix_names{i};
    for j = 1:length(order)
        idx = order(j);
        method_name = method_names{idx};
        load(sprintf('../data/%s_%s.mat', matrix_name, method_name));
        
            errors = trace_norm_errors(1:last_idx,:);
            
            figure(i)
            if ~strcmp(method_name, 'Greedy')
                errorbar(ks, mean(errors, 2), std(errors, 1, 2),...
                    specs{idx}, 'LineWidth', 2, 'Color', colors{idx},...
                    'MarkerFaceColor',colors{idx})
            else
                plot(ks, mean(errors, 2), specs{idx}, 'LineWidth', 2,...
                    'Color', colors{idx})
            end
            hold on
    end
    
    figure(i)
    if strcmp(matrix_name, 'smile')
        legend(method_names_for_plot{order}, 'location', 'best','FontSize',26)
    end
    xlabel('Rank $k$','FontSize', 26)
    ylabel('Relative Trace Error', 'FontSize', 26)
    set(gca, 'YScale', 'log')
    ax = gca; ax.FontSize = 20; 
    saveas(gcf, sprintf('../figs/%s.png', matrix_name));
end

